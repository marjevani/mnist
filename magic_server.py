from backend import Inference_manager
from http import server
import socketserver
import logging
import json

PORT = 80


class ServerHandler(server.SimpleHTTPRequestHandler):

    def __init__(self, request, client_address, server):
        self.manager = server.manager
        self.manager.server_use = self
        super().__init__(request, client_address, server)
    # 
    def do_POST(self):
        print("got POST req")
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\nBody:\n%s\n",
                     str(self.path), str(self.headers), post_data.decode('utf-8'))

        post_msg = json.loads(post_data)

        if post_msg["filter"]:
            self.manager.filter = post_msg["filter"]
        if post_msg["power"]:
            self.manager.power = post_msg["power"]

        print("incoming post fields:")
        print("filter =\n" + str(post_msg["filter"]))
        print("power = \n" + str(post_msg["power"]))
        print("pic len = " + str(len(post_msg["data"])))

        try:
            self.process_img = self.manager.pre_process(post_msg["data"])
            self.manager.send_eval(None)
        except Exception as e:
            self.process_img = 0
            error = "ERROR accured in the server" + str(e)
            print(error)
            self.send_result(error, 0)

    def send_result(self, res, stats):
        # print("sending img: \n" + str(self.process_img))
        data_json = json.dumps({"data": res, "img": self.process_img, "stats": stats})

        print("final result = " + res)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(data_json.encode())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Handler = ServerHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print("Building Deep Learning Net:")
    httpd.manager = Inference_manager.Inference_manager(server=True)
    print("serving at port", PORT)
    httpd.serve_forever()
