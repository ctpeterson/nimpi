import nimpi

mpi:
  assert size() >= 2, "This example requires at least 2 processes."

  let comm = WorldCommunicator
  let r    = comm.myRank
  let n    = comm.size

  # send/recv: rank 0 sends to rank 1, rank 1 receives from rank 0.

  var arraySend = [1, 2, 3, 4, 5]
  var arrayRecv = newSeq[int](5)

  if r == 0:
    echo "Rank 0 sending: ", arraySend
    comm.send(arraySend, 1, tag = 1)
  if r == 1:
    discard comm.receive(arrayRecv, 0, tag = 1)
    echo "Rank 1 received: ", arrayRecv

  comm.mpiBarrier()

  # sendReceive: each rank exchanges with its ring neighbors using separate
  # send and receive buffers. MPI_Sendrecv handles scheduling internally so
  # there is no risk of deadlock regardless of the number of ranks.

  var srSend = [r * 10 + 0, r * 10 + 1, r * 10 + 2, r * 10 + 3, r * 10 + 4]
  var srRecv = newSeq[int](5)
  let srSt = comm.sendReceive(
    srSend,
    srRecv,
    dest   = (r + 1) mod n,
    source = (r - 1 + n) mod n
  )
  echo "Rank ", r, " sendReceive: received from rank ", srSt.source, ": ", srRecv

  comm.mpiBarrier()

  # sendReceiveReplace: same ring exchange but in-place — the buffer is
  # overwritten with the incoming data, so no separate receive buffer is needed.

  var ringBuf = [r * 10, r * 10 + 1, r * 10 + 2]
  echo "Rank ", r, " before ring shift: ", ringBuf
  discard comm.sendReceiveReplace(
    ringBuf,
    dest   = (r + 1) mod n,
    source = (r - 1 + n) mod n
  )
  echo "Rank ", r, " after ring shift: ", ringBuf

  comm.mpiBarrier()

  # non-blocking send/receive: post both operations immediately, allowing other
  # work to proceed, then wait for both to complete. Each rank sends to its
  # right neighbor and receives from its left neighbor in the ring.

  var nbSend = [r * 100 + 0, r * 100 + 1, r * 100 + 2]
  var nbRecv = newSeq[int](3)

  var sendReq = comm.immediateSend(nbSend, dest   = (r + 1) mod n)
  var recvReq = comm.immediateReceive(nbRecv, source = (r - 1 + n) mod n)

  # ... other work could go here ...

  discard sendReq.wait
  let nbSt = recvReq.wait
  echo "Rank ", r, " non-blocking recv: received from rank ", nbSt.source, ": ", nbRecv