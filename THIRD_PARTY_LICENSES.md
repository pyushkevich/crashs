# Third-Party Code and Licenses

## cbstools-public (CBS Tools)

CRASHS's cortical reconstruction pipeline (`run_cruise()` in `src/crashs/crashs.py`)
calls into native code compiled from **cbstools-public**, the CBS Tools Java library
developed by Pierre-Louis Bazin and colleagues at the Max Planck Institute for Human
Cognitive and Brain Sciences (MPI CBS), Leipzig, Germany:

- Upstream repository: https://github.com/piloubazin/cbstools-public
- Vendored here as a pinned git submodule at `native/cbstools-public/` (currently
  commit `27a5bfeb827de75a0b55f97ed8cf57a5b7cc8b11`, branch `master`) — the original
  source is not copied or modified, only compiled.
- License: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**.
  The full license text as distributed with cbstools-public is reproduced verbatim
  below and also travels with the submodule at `native/cbstools-public/LICENSE.md`.

**What CRASHS adds, and what it doesn't change**: the five cbstools-public classes
CRASHS uses (`ShapeTopologyCorrection2`, `CortexOptimCRUISE`, `SurfaceLevelsetToMesh`,
`SurfaceInflation`, `LaminarVolumetricLayering`, all under
`native/cbstools-public/de/mpg/cbs/`) are used **unmodified**. CRASHS adds a thin Java
wrapper layer, under `native/java/src/de/mpg/cbs/crashs/`, that exposes each class's
existing public methods through a C-callable API (via GraalVM Native Image
`@CEntryPoint`s), so the algorithms can be AOT-compiled into a native shared library
and called from Python without a JVM at runtime. This wrapper layer contains no
scientific/algorithmic logic of its own — it only marshals data across the native
boundary — and is CRASHS's own code (MIT-licensed, see `LICENSE`).

The compiled native library that CRASHS ships (`crashs/_native_lib/libcbstools_native.*`,
built from `native/cbstools-public` plus the wrapper layer above) is a **derivative
work of cbstools-public** and is distributed under the terms of CC BY-SA 4.0, in
addition to CRASHS's own MIT license covering the rest of the codebase. See the
`Additional attribution and ShareAlike terms` note in `LICENSE` for how the two
licenses apply to different parts of this repository.

Relevant background/citation for the CRUISE cortical reconstruction method used here:

> Han X, Pham DL, Tosun D, Rettmann ME, Xu C, Prince JL. CRUISE: cortical
> reconstruction using implicit surface evolution. NeuroImage. 2004;23(3):997-1012.
> https://doi.org/10.1016/j.neuroimage.2004.06.043

### cbstools-public LICENSE.md (verbatim)

Copyright (c) 2016 Max Planck Institute for Human Cognitive and Brain Sciences

This software is copyrighted by the Max Planck Institute for Human Cognitive
and Brain Sciences, Leipzig, Germany and licensed under a Creative Commons
Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
The following terms also apply to all files associated with the software
unless explicitly disclaimed in individual files.

The authors hereby grant permission to use, copy, and distribute this software
and its documentation for any purpose, provided that existing copyright notices
are retained in all copies and that this notice is included verbatim in any
distributions. Additionally, the authors grant permission to modify this
software and its documentation for any purpose, provided that such modifications
are distributed with explicit attribution of the original authors and description
of the changes, and that existing copyright notices are retained in all copies.

IN NO EVENT SHALL THE AUTHORS OR DISTRIBUTORS BE LIABLE TO ANY PARTY FOR DIRECT,
INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF
THIS SOFTWARE, ITS DOCUMENTATION, OR ANY DERIVATIVES THEREOF, EVEN IF THE AUTHORS
HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE AUTHORS AND DISTRIBUTORS SPECIFICALLY DISCLAIM ANY WARRANTIES, INCLUDING, BUT
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE, AND NON-INFRINGEMENT. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, AND
THE AUTHORS AND DISTRIBUTORS HAVE NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT,
UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Full CC BY-SA 4.0 legal code: https://creativecommons.org/licenses/by-sa/4.0/legalcode

## Other bundled files

- `native/java/lib/commons-math3-3.5.jar` — Apache Commons Math, Apache License 2.0
  (https://commons.apache.org/proper/commons-math/). Vendored unmodified as a compile-time
  dependency of cbstools-public; not itself modified or redistributed with changes.
- `native/java/lib/Jama-mipav.jar` — JAMA (Java Matrix Package), public domain, as
  redistributed by the MIPAV project. Vendored unmodified as a compile-time dependency
  of cbstools-public.
