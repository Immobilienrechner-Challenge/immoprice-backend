docker build . -t immoprice-backend
docker run -p 8099:8099 --name immoprice-backend immoprice-backend