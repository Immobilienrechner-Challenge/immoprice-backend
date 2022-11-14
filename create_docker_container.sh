docker build . -t immoprice-backend
docker run -p 8099:8099 --name immoprice-backend --network nginxproxymanager_default immoprice-backend 