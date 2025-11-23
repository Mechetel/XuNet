curl -L -o ~/data/GBRASNET.zip https://www.kaggle.com/api/v1/datasets/download/zapak1010/bossbase-bows2
unzip ~/data/GBRASNET.zip -d ~/data/
rm ~/data/GBRASNET.zip

cp -r ~/data/GBRASNET/BOSSbase-1.01 ~/data/GBRASNET/BOSSbase-1.01-div

touch ~/data/split.sh
sh ~/data/split_server.sh
