# Documentation

For all documentation on all-screen text: use ASD-STE100 Simplified Technical English.
Use active voice.
Limit each sentence to 20 words.
State one fact per sentence.

## Audience and tone

Assume readers understand Linux, Docker, SSH, and distributed model serving.
Respect their judgment.
Use neutral technical language, not lectures or permission reminders.
Omit obvious cautions about stopping workloads, concurrent engines, approvals, reboots, and credential hygiene.
Document non-obvious recipe constraints and their consequences.
Prefer a specific fact over a prohibition.
Use direct instructions for recipe steps, not routine operator responsibilities.

## Present state

Describe only the current supported state.
Do not narrate migrations, earlier versions, failed attempts, or debugging sessions.
Explain what to do, not how the project arrived here.
State recipe-specific hazards as current facts, not incident stories.
Keep benchmark setup, scores, and limits in results documents.
Report unsuccessful evaluation outcomes without a troubleshooting narrative.

## Brevity

Write for an operator who wants to run a recipe.
Lead with the required action or decision.
Delete filler, repeated facts, obvious statements, and descriptions of unchanged behavior.
Do not restate commands or table values in prose.
State each recipe-specific constraint once, near the relevant step.

## Structure

Target 300 words per README, excluding code and data tables.
Use more only for essential setup or safety requirements.
Use numbered steps for required sequences.
Use tables for settings and measurements, not paragraphs.
Link to shared instructions instead of copying them.
Omit implementation details unless they affect operation, compatibility, safety, or interpretation of results.
Do not add files or tables merely to hide word count.

## Review

Delete any sentence whose removal changes no decision, action, or interpretation.
Compare prose word counts before and after editing.
Preserve commands, current settings, recipe-specific constraints, benchmark measurements, and attribution.
Remove generic cautions instead of softening their wording.
Do not remove essential facts to meet length targets.
Keep verbatim license text and copyright notices unchanged.
