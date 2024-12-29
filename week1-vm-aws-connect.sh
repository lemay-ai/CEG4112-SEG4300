icacls uo-aws.pem /inheritance:r
icacls uo-aws.pem /grant:r "$($env:USERNAME):(R)"
icacls uo-aws.pem /remove "Authenticated Users" "BUILTIN\textbackslash Users" "BUILTIN\textbackslash Administrators"
ssh -i uo-aws.pem -L 8888:localhost:8888 ubuntu@3.99.169.227
docker pull tensorflow/tensorflow:2.17.0-gpu-jupyter
sudo bash
docker run --gpus all -it -d --name tensorflow_uottawa -p 8888:8888 tensorflow/tensorflow:2.17.0-gpu-jupyter
ps ax|grep jupyter
docker ps
docker exec -it f5795d69b4f5 bash
jupyter notebook list