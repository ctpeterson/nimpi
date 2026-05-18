import nimpi

mpi:
  assert WorldCommunicator.size == 4

  var vectorData1: array[5, int]
  var vectorData2 = newSeq[int](5)

  # broadcast: vector data

  if myRank() == 0:
    for i in 0..<5: (vectorData1[i], vectorData2[i]) = (i, i)
  WorldCommunicator.broadcast(vectorData1, 0)
  if myRank() != 0: 
    echo "Process ", myRank(), " received data: ", vectorData1

  # scatter
  
  var sendData: array[8, int]
  var recvData: array[2, int]
  if myRank() == 0:
    for i in 0..<8: sendData[i] = i
  WorldCommunicator.scatter(sendData, recvData, 0, sendData.len div WorldCommunicator.size, recvData.len)
  if myRank() != 0:
    echo "Process ", myRank(), " received data: ", recvData

  # gather

  var gatherSendData: array[2, int]
  var gatherRecvData: array[8, int]
  for i in 0..<2: gatherSendData[i] = myRank() * 10 + i
  WorldCommunicator.gather(gatherSendData, gatherRecvData, 0, gatherSendData.len, gatherRecvData.len div WorldCommunicator.size)
  if myRank() == 0:
    echo "Process ", myRank(), " gathered data: ", gatherRecvData
  
  # reduce

  var reduceSendData: array[5, int]
  var reduceRecvData: array[5, int]
  for i in 0..<5: reduceSendData[i] = myRank() * 10 + i
  WorldCommunicator.reduce(ReduceSum, reduceSendData, reduceRecvData, 0, reduceSendData.len, reduceRecvData.len)
  if myRank() == 0:
    echo "Process ", myRank(), " reduced data: ", reduceRecvData