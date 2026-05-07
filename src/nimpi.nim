#[
  NiMPI: https://github.com/ctpeterson/nimpi
  Source: src/nimpi.nim
  Author: Curtis Taylor Peterson <curtistaylorpetersonwork@gmail.com>

  Acknowledgments:
    - NimMPI by Michalina Kotwica (Udiknedormin)
      https://github.com/Udiknedormin/NimMPI — MIT License
      Copyright (c) 2016 M. Kotwica
    - QEX (Quantum EXpressions) by James Osborn et al.
      https://github.com/jcosborn/qex — MIT License
      Copyright (c) 2015 James Osborn

  MIT License

  Copyright (c) 2026 Curtis Taylor Peterson

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
]#

import nimpi/[mpi]

export mpi

## Nim MPI bindings and utilities. This module provides a high-level interface to MPI,
## including type definitions, global variables, and utility procedures for initialization,
## finalization, and error handling. It serves as the main entry point for users of the
## NiMPI library, abstracting away the details of the underlying MPI implementation and
## providing a more idiomatic Nim interface to MPI functionality.
## 
## The actual MPI function bindings are defined in the `mpiwrap` module.
## 
## The idiomatic Nim interface is provided by the `mpi` module. 
## 
## Acknowledgments:
## - NimMPI by Michalina Kotwica (Udiknedormin)
##  https://github.com/Udiknedormin/NimMPI — MIT License
##  Copyright (c) 2016 M. Kotwica
## - QEX (Quantum EXpressions) by James Osborn et al.
##  https://github.com/jcosborn/qex — MIT License
##  Copyright (c) 2015 James Osborn
##
## Example:
## ```nim
## import nimpi
## 
## mpi:
##   echo "Hello from process ", WorldCommunicator.myRank, " of ", WorldCommunicator.size
## ```