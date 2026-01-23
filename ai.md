DendryTerra is a plugin for Terra to generate Dendry noise.

## Design outline

Functions are intended to take advantage of Seismic (https://github.com/PolyhedralDev/Seismic) for efficiency in java.

Dendry's implementation is based on a forked copy (https://github.com/otto-link/Noise)

## Theory

Dendry noise construction is detailed in "Gaillard19I3D.pdf"

## Building

DendryTerra is built using gradlew, and can be triggered by calling gradlew.bat in windows environments to gather dependencies and build the final addon jar file.

## Installation and use

The final jar file should be placed in the addons folder of the related Terra tool (Terra, NoiseTool, BiomeTool).  After starting the corresponding Terra tool, the console / log will indicate the DendryTerra addon loaded successfully along with it's contributors.  After this, the dendry sampler can be called like any other sampler:

sampler:
    type: DENDRY
    n:  6
    epsilon: 0
    delta: 0.075
    slope:
    frequency: 0.001
    return: elevation
    sampler: *PointerToControlFunction

Where:
- n is the number of branches, generally recommended not to exceed 3
    Default value: 2
- epsilon is the randomness factor from 0 to 0.5, where 0.5 pushes the q-point to the center of the evaluation cell.
    Default value: 0
- delta (from 0 +, not recommended to exceed 0.08 or else branch overlaps can occur) influences the branching behavior - it increases the randomization of a segment by subdividing it into equal parts and perturbing the distance of the intermediate points to the initial segment by a random value.
    Default value: 0.05
- slope - affect valley / cliff generation.
    Default value: 0.5
- frequency may not be necessary, but is a placeholder to define the initial grid size.  This may be able to be replaced with another parameter depending on the implementation of the original algorithm.
    Default value: 0.001
- return - a string that indicates the value that is returned from the sampler for the x/z sample.
    - "distance" returns the euclidean distance to the closest branch.
    - "weighted" returns the weighted distance to the "closest" branch, where the distance is weighted according to the branch level.  The weighted distance is the true distance to a branch multiplied by the branch level, so that iterations higher up the sampler are weighted less heavily.
    - "elevation" returns the elevation difference from the root branch.
    Default value: elevation.
- sampler is a pointer to the control function that influences the overall shape of the structure, typically this would be elevation if this filter is being used to plant rivers on top of a topology.

## Code expectations
The code is expected to follow the Terra plugin conventions, and utilize Seismic for optimized math libraries where applicable.

Some of the addons for Terra and there dependencies are available here: "https://maven.solo-studios.ca/#/releases/com/dfsek", and may have application to this project.

### Notes for "AI" (Machine learned / LLM) assistants

1. **Scaffold & Prototype**
 
   * Generate starter projects, boilerplates, module skeletons, Terraform modules, Dockerfiles, GitHub Actions workflows, etc.
   * Ask clarifying questions before scaffolding anything major (e.g. frameworks, directory structure, cloud provider).
 
2. **Implement & Refine**
 
   * Write clean, idiomatic TypeScript (or other requested language) with inline comments and clear variable names.
   * Make use of the logging mechanisms in the Terra API to report unexpected behavior, and / or create log files in the same directory as the addon jar file with log information from the latest run (replace the log file each run)
   * Adhere to best practices around modularity, error handling, and security.
 
3. **Document & Explain**
 
   * Provide concise, step‑by‑step instructions for any setup or deployment.
   * Embed helpful comments and docstrings throughout generated code, point to references when relevant.
   * When introducing new concepts (e.g. a Terraform provider), include a 1–2 sentence definition.
 
4. **Test & Validate**
 
   * Generate unit and integration tests (e.g. Jest, Mocha) alongside code.
   * Include sample test data and commands to run them.
   * Suggest CI steps (GitHub Actions) to automate tests and linting.
 
5. **Troubleshoot & Debug**
 
   * When errors arise, analyze logs or stack traces and propose 2–3 probable root causes plus concrete fixes.
   * Offer alternative workarounds if the primary solution is blocked.
 
6. **Simplicity & Iteration**
 
   * **Keep It Simple**: Avoid unnecessary complexity that could break basic functionality.
   * **Build Incrementally**: After each new feature or module, verify core workflows end‑to‑end before layering in more.
   * **Yank, Don’t Yank Too Much**: If you see code getting tangled, suggest targeted refactors rather than wholesale rewrites.
   * **Smoke Tests**: Start with a minimal “hello world” or basic use‑case test to confirm nothing’s broken before adding bells and whistles.  This is especially important when working with the Terra API.
 
---
 
#### Style & Format Guidelines
 
* **Clarity First**: Short sentences, minimal jargon, **bold** key commands or config snippets.
* **Step‑by‑Step**: Use numbered lists. Each step must be a standalone action.
* **Code Blocks**: Wrap code in fenced blocks with language tags.
* **Ask Before You Leap**: If any assumption is unclear (e.g. target Node version, cloud region), request clarification.
* **Encouraging Tone**: Be supportive, forward‑thinking, and sprinkle in quick, clever humor (e.g., “Let’s squash this bug like it owes us money!”).
* **Accessibility**: Offer extra background for beginners, but clearly label “Advanced Tips” sections for experts.
 
---
  
#### Best Practices & Security
 
* Validate inputs and fail fast.
* Encourage automated linting (ESLint/Prettier) and type‑checking.

#### Initial code generation

* Start with a minimal viable implementation that fulfills core requirements.
* This should include gradel.bat, all gradel related build files / configurations and dependency pulls.
* Ideally compatible with Java 23 and built with Gradle 10
* This is a terra plugin, additional context can be referenced at the following locations:

- "C:\Projects\Terra\common\addons\config-noise-function\src\main\java\com\dfsek\terra\addons\noise" contains Terra and the path to the config-noise addon that might be a good reference.
** 
- "C:\Projects\NoiseTool" contains the noise tool and it's source code that -like other Terra inclusive tools- will be capable of loading this addon.
- "https://terra.polydev.org/api/index.html" contains indexes around the Terra API.  This information is outdated but likely still relevant.
- "C:\Projects\DendryNoise" contains the source code to the initial draft to simulate Dendry noise using CPP, but for java this may need to be reverse engineered and reformed for java compatible bytecode.

## License

See the LICENSE file.

