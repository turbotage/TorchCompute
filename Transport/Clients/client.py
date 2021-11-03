

import sys
sys.path.insert(1, '../../build/Transport/gRPC/')

import grpc

import Transport_pb2
import Transport_pb2_grpc

import io
import torch

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        request = Tensor_pb2.TensorRequest(tensor_name=b"T1")

        stub = Tensor_pb2_grpc.TensorSenderStub(channel)
        responses = stub.SendTensor(request)

        tensor_str = b""
        for  response in responses:
            tensor_str += response.tensor_str

        print("tensor_str: ", tensor_str)

        f = io.BytesIO(tensor_str)
        
        tensor = torch.load(f)

        print(tensor)

if __name__ == '__main__':
    run()
