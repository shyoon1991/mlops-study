import warnings; warnings.filterwarnings("ignore")
import socket, mlflow
from mlflow.store.artifact import http_artifact_repo as h
mlflow.set_tracking_uri("http://localhost:5000")
orig = h.HttpArtifactRepository._multipart_download
def spy(self, presigned_response, remote_file_path, *a, **k):
    print("presigned URL host:", presigned_response.url.split("/")[2], "| file:", remote_file_path, flush=True)
    raise SystemExit(0)
h.HttpArtifactRepository._multipart_download = spy
try:
    socket.gethostbyname("seaweedfs"); print("seaweedfs resolves")
except Exception as e:
    print("client resolve 'seaweedfs':", type(e).__name__)
import mlflow.sklearn
mlflow.sklearn.load_model("models:/random-forest-classifier/2")
