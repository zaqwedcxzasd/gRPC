import time
import grpc
from proto import stream_pb2
from proto import stream_pb2_grpc

def generate_requests():
    """
    リクエストを生成するイテレータ（ジェネレータ）関数。
    これが非同期にgRPCによって読み取られ、サーバーに送信されます。
    """
    for i in range(1, 51): # 50回送信
        print(f"Client: 送信 ID={i}, Value={i}")
        request = stream_pb2.Request(id=i, value=i)
        yield request
        time.sleep(0.05) # 少しウェイトを入れて送信の様子をわかりやすくする

def run():
    print("Client: サーバーに接続中...")
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = stream_pb2_grpc.StreamServiceStub(channel)
        
        print("Client: 双方向ストリーミング開始")
        
        # スタブのメソッドにイテレータを渡すと、送信が始まります。
        # 戻り値として、サーバーからのレスポンスを受け取るイテレータが返ってきます。
        # これにより、送信(generate_requests)と受信(responsesのループ)が並行して動いているように見えます。
        responses = stub.BidirectionalStream(generate_requests())
        
        try:
            for response in responses:
                print(f"Client: 受信 Message='{response.message}', Count={response.count}, Sum={response.sum}")
        except grpc.RpcError as e:
            print(f"Client: エラーが発生しました: {e}")

if __name__ == '__main__':
    run()
