from concurrent import futures
import time
import grpc
from proto import stream_pb2
from proto import stream_pb2_grpc

class StreamService(stream_pb2_grpc.StreamServiceServicer):
    def BidirectionalStream(self, request_iterator, context):
        """
        双方向ストリーミングの実装。
        クライアントからのリクエストをイテレータとして受け取ります。
        """
        buffer = []
        
        print("Server: ストリーム開始")
        
        # request_iteratorからデータを順次取り出す
        for request in request_iterator:
            print(f"Server: 受信 ID={request.id}, Value={request.value}")
            
            # 1. データをバッファリング
            buffer.append(request.value)
            
            # 2. 受信データが10個溜まったかチェック
            if len(buffer) >= 10:
                print(f"Server: バッファが10個溜まりました。処理を実行します。: {buffer}")
                
                # 合計値を計算 (処理のシミュレーション)
                total_sum = sum(buffer)
                
                # 3. 閾値チェック (例として合計がこれ以上ならイベントなど)
                # 今回は単純に10個溜まったら必ず応答を返す実装にします
                
                response = stream_pb2.Response(
                    message="バッファが一杯になりました",
                    count=len(buffer),
                    sum=total_sum
                )
                
                # クライアントへレスポンスを返す (yield)
                yield response
                
                # バッファをクリア
                buffer = []
                
        print("Server: ストリーム終了")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    stream_pb2_grpc.add_StreamServiceServicer_to_server(StreamService(), server)
    server.add_insecure_port('[::]:50051')
    print("Server: ポート 50051 で待機中...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
