import nimpi

mpi:
  assert size() >= 2, "This example requires at least 2 processes."

  # send/recv scalars

  var scalarSend = 42
  var scalarRecv = 0

  if myRank() == 0:
    echo "Rank 0 sending: ", scalarSend
    scalarSend.send(1, tag = 0)
  if myRank() == 1:
    scalarRecv.receive(0, tag = 0)
    echo "Rank 1 received: ", scalarRecv

  # send/recv arrays

  var arraySend = [1, 2, 3, 4, 5]
  var arrayRecv = newSeq[int](5)

  if myRank() == 0:
    echo "Rank 0 sending: ", arraySend
    arraySend.send(1, tag = 1)
  if myRank() == 1:
    arrayRecv.receive(0, tag = 1)
    echo "Rank 1 received: ", arrayRecv