# SPDX-License-Identifier: Apache-2.0
"""SparkRun 0.3.6 runtime for this pinned, switched, four-Spark deployment."""

import shlex

from sparkrun.orchestration.primitives import build_ssh_kwargs
from sparkrun.orchestration.ssh import run_remote_script
from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.utils.text import render_template


class TensorFoldDSV41Runtime(RuntimePlugin):
    runtime_name = "tensorfold-dsv41"
    requires_capability = frozenset({"gb10"})

    def cluster_strategy(self):
        return "native"

    def generate_command(
        self,
        recipe,
        overrides,
        is_cluster,
        num_nodes=1,
        head_ip=None,
        skip_keys=frozenset(),
    ):
        return self.generate_node_command(recipe, overrides, head_ip, num_nodes, 0)

    def generate_node_command(
        self,
        recipe,
        overrides,
        head_ip,
        num_nodes,
        node_rank,
        init_port=29760,
        skip_keys=frozenset(),
        hosts=None,
        placement=None,
    ):
        cfg = recipe.build_config_chain(overrides)
        if num_nodes != 4 or int(cfg.get("tensor_parallel")) != 4:
            raise ValueError("This TensorFold recipe requires exactly four TP ranks")
        if not 0 <= node_rank < 4:
            raise ValueError("TensorFold rank must be between 0 and 3")
        if int(cfg.get("pool")) < int(cfg.get("max_model_len")):
            raise ValueError("The shared pool must hold at least one full context")
        cfg.set("node_rank", node_rank)
        cfg.set("draft_flag", "" if int(cfg.get("drafts")) else "--no-drafts")
        template = (
            recipe.command
            if node_rank == 0
            else recipe.runtime_config["worker_command"]
        )
        command = render_template(template.strip(), cfg)
        # One config chain drives both the head's environment and follower flags.
        settings = {
            "TF_DS_WINDOW": "pool",
            "TF_DS_STREAMS": "max_num_seqs",
            "TF_DS_ROUND_ROWS": "round_rows",
            "TF_DS_GRAPHS": "graphs",
            "TF_DS_HC_SIDE": "side",
            "TF_DS_L2_PREFETCH": "prefetch",
            "TF_DS_ENGRAM_AIO": "aio",
        }
        prefix = shlex.join(
            ["env", *(f"{k}={cfg.get(v)}" for k, v in settings.items())]
        )
        return f"{prefix} {command}"

    def prepare(
        self,
        recipe,
        hosts,
        config=None,
        dry_run=False,
        transfer_mode="auto",
        overrides=None,
    ):
        if hosts != recipe.runtime_config["hosts"]:
            raise ValueError(
                "Use the four hosts in runtime_config.hosts, in rank order"
            )
        if dry_run:
            return
        cfg = recipe.build_config_chain(overrides)
        key = shlex.quote(str(cfg.get("api_key_file")))
        # Fail before SparkRun's cache-clearing/launch phase. Never evict an
        # existing inference job or try to reclaim memory during serving.
        for rank, host in enumerate(hosts):
            check = (
                "set -eu\nawk '/MemAvailable:/ {exit ($2 < 104857600)}' /proc/meminfo\n"
            )
            if rank == 0:
                check += f"test -f {key} && test -s {key}\n"
            result = run_remote_script(
                host, check, timeout=30, **build_ssh_kwargs(config)
            )
            if not result.success:
                raise RuntimeError(
                    f"{host}: need 100 GiB available (and the head's existing API key); "
                    "stop workloads and reclaim memory only in an approved maintenance window"
                )

    def _run_cluster(
        self,
        hosts,
        image,
        serve_command="",
        recipe=None,
        overrides=None,
        *,
        cluster_id="sparkrun0",
        env=None,
        cache_dir=None,
        config=None,
        dry_run=False,
        detached=True,
        trust=False,
        **kwargs,
    ):
        from sparkrun.runtimes._cluster_ops import (
            ClusterContext,
            _attach_foreground,
            cleanup_after_failure,
            cleanup_ranked_containers,
            exec_serve_on_container,
            run_pre_serve_hooks,
        )

        cfg = recipe.build_config_chain(overrides)
        ctx = ClusterContext.build(
            runtime=self,
            hosts=hosts,
            image=image,
            cluster_id=cluster_id,
            env=env,
            cache_dir=cache_dir,
            config=config,
            dry_run=dry_run,
            recipe=recipe,
            placement=kwargs.get("placement"),
        )
        executor = self._resolve_executor()
        # Only these job-specific SparkRun names are cleaned up. The existing
        # dsv41-1017-tf containers are never adopted, stopped or renamed here.
        if cleanup_ranked_containers(ctx, executor):
            return 1
        containers = [
            (host, executor.node_container_name(cluster_id, rank))
            for rank, host in enumerate(hosts)
        ]
        for rank, (host, name) in enumerate(containers):
            volumes = {str(cfg.get("hf_cache")): "/hf:ro"}
            if rank == 0:
                volumes[str(cfg.get("api_key_file"))] = "/run/secrets/model-api-key:ro"
            script = executor.generate_launch_script(
                image=image,
                container_name=name,
                command="sleep infinity",
                env=ctx.all_env,
                volumes=volumes,
                extra_docker_opts=kwargs.get("extra_docker_opts"),
                sparkrun_labels=executor.workload_labels_for_cluster(
                    cluster_id=cluster_id, recipe=recipe, runtime=self, rank=rank
                ),
            )
            result = run_remote_script(
                host, script, timeout=120, dry_run=dry_run, **ctx.ssh_kwargs
            )
            if not result.success and not dry_run:
                cleanup_after_failure(
                    ctx, executor, reason=f"rank {rank} container launch failed"
                )
                raise RuntimeError(result.stderr)
        run_pre_serve_hooks(
            self, ctx, containers, recipe, overrides, trust=trust, cache_dir=cache_dir
        )
        # Match the working launcher: followers 3,2,1, then head. Do NOT use
        # SparkRun's generic native TCP readiness probe: connecting to a
        # TensorFold rendezvous port consumes its one real follower socket.
        for rank in (3, 2, 1, 0):
            host, name = containers[rank]
            command = self.generate_node_command(recipe, overrides, None, 4, rank)
            if exec_serve_on_container(ctx, executor, host, name, command):
                cleanup_after_failure(ctx, executor, reason=f"rank {rank} serve failed")
                return 1
        self._print_connection_info(hosts, cluster_id, per_node_logs=True)
        if not detached and not dry_run:
            _attach_foreground(self, ctx, kwargs.get("follow", True))
        return 0

    def _stop_cluster(self, hosts, cluster_id, config=None, dry_run=False):
        return self._stop_native_cluster(
            hosts, cluster_id, config=config, dry_run=dry_run
        )
