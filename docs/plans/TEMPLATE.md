<!--
  devai plan template.

  Usage:
    cp docs/plans/TEMPLATE.md docs/plans/<short-kebab-title>.md
    Then fill in each section. Delete any section that genuinely does not
    apply (e.g. a single-phase plan does not need "Phase 2"). Do NOT keep
    empty sections around for cosmetic reasons -- a missing section is
    clearer than an empty one.

  Style:
    - ASCII only (see CLAUDE.md "Documentation conventions"): -- not em-dash,
      -> not arrow, "..." not ellipsis, straight quotes only.
    - Be specific. Plans are read by people who were not in the room when
      the decisions were made. State the why, not just the what.
    - Link to prerequisite plans by relative path:
      `[Plan: foo](./foo.md)`. Do not hardcode commit SHAs.
-->

# <Plan Title>

<!-- One-line goal. Should be readable in isolation. Example:
     "Add NATS-based fleet coordination plane for multi-host devai workers." -->

_<one-line goal>_

## Status

<!-- One of: Draft / Approved / In Progress / Done / Cancelled / Superseded.
     Optionally followed by a date or a short note. -->

Draft. Not yet scheduled for execution.

## Dependencies

<!-- List prerequisite plans that must reach "Done" before this plan can
     start. Use relative links. If a dependency is not yet a written plan,
     write a placeholder line describing what it is, so a future reader
     understands the gap.

     If there are no dependencies, write "None." Do not leave this section
     empty -- explicit "None" is the signal that the author considered it. -->

- [Plan: <name>](./<file>.md) -- <one-line reason this is a prerequisite>
- (placeholder) <description of an unwritten prerequisite plan>

## Enables / Unblocks

<!-- What downstream work becomes possible when this plan lands. Used by
     reviewers to judge priority. Be concrete: name the features or plans
     this unblocks, not just "future flexibility". -->

- <downstream capability or plan this enables>
- <...>

## Out of scope

<!-- Explicit non-goals. Anything a reasonable reader might assume is in
     scope but is not. This section prevents scope creep during execution
     and gives reviewers something to disagree with up front. -->

- <thing that sounds related but is not part of this plan>
- <...>

## Open questions

<!-- Decisions parked for explicit answer BEFORE execution starts. Each
     entry should be a question, not a statement, and should be
     answerable with a single choice. If there are no open questions,
     write "None." -->

1. <question> -- recommendation: <option>.
2. <question> -- recommendation: <option>.

## Context

<!-- Why this plan exists. The problem being solved, the constraint that
     forces this approach, the prior art that was considered and rejected.
     Roughly 1-3 paragraphs. A reader unfamiliar with the recent
     conversation should be able to understand the rationale here. -->

<background paragraph(s)>

## Approach

<!-- The shape of the solution in one paragraph. NOT the detailed steps --
     those go in the Phase sections below. This paragraph should let a
     reviewer agree or disagree with the overall direction before they
     read the implementation detail. -->

<one-paragraph summary of the solution shape>

---

## Phase 1 -- <short name>

<!-- For multi-phase plans, each phase should be independently shippable:
     a phase that requires another phase to be useful is not a phase, it
     is a step. If the plan is single-phase, delete the Phase headings and
     promote the content to top level under "Implementation".

     Each phase should have: a goal, a deliverables list, detailed steps,
     exit criteria, and risks specific to that phase. -->

### Goal

<what this phase achieves on its own>

### Deliverables

```
<file or directory>             <new | modify> -- <brief description>
<...>
```

### Detailed steps

1. <step>
2. <step>
3. <...>

### Exit criteria

- <observable, testable condition that says "this phase is done">
- <...>

### Phase 1 risks

| Risk                       | Mitigation                          |
| -------------------------- | ----------------------------------- |
| <risk>                     | <mitigation>                        |

---

## Phase 2 -- <short name>

<!-- Repeat the Goal / Deliverables / Detailed steps / Exit criteria / Risks
     pattern for each additional phase. Delete this whole section if the
     plan is single-phase. -->

...

---

## Combined risk register

<!-- Cross-phase risks, or a consolidated view of all risks if the plan is
     large. For small plans, the per-phase risk tables are sufficient and
     you can delete this section. -->

| Risk                       | Phase | Mitigation                          |
| -------------------------- | ----- | ----------------------------------- |
| <risk>                     | <n>   | <mitigation>                        |

## Migration / rollback story

<!-- How to undo this plan if it goes wrong, and what existing users
     experience during the transition. For a new feature with no existing
     users, "rollback = revert the PR" is a valid answer -- write it
     explicitly so reviewers know you considered it. -->

- <how to back out>
- <upgrade path for existing installs, if any>

## Estimated effort

| Phase    | Engineering effort           | Wall-clock     |
| -------- | ---------------------------- | -------------- |
| Phase 1  | <PR count + LoC ballpark>    | <days>         |
| Phase 2  | <...>                        | <...>          |
| Total    | <PR count>                   | <total time>   |

## References

<!-- Upstream docs, prior art, related plans, relevant book chapters.
     Plain text references (project name + URL fragment) are fine; this
     is not a citations page. -->

- <reference>
- <reference>
