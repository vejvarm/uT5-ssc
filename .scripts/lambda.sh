
# on Nemesis
cd /work/lambda
tar -czvf ut5-ep19-dirty-v2.tgz /work/results/t5/ut5-ep19-openwebtext-dirty_v2/final_dirtymatched_grouped
tar -czvf ut5-ep19-injected-10p.tgz /work/results/t5/ut5-ep19-openwebtext-injected-sparql-10p/final_clean_grouped
tar -czvf ut5-ep19-injected-30p.tgz /work/results/t5/ut5-ep19-openwebtext-injected-sparql-30p/final_clean_grouped

scp -i ~/.ssh/freya.pem uT5-repo-clean.tgz ubuntu@192.222.56.220:~/git/
scp -i ~/.ssh/freya.pem /work/lambda/ut5-ep19-clean.tgz ubuntu@192.222.56.220:~/data/
scp -i ~/.ssh/freya.pem /work/lambda/ut5-ep19-dirty-v2.tgz ubuntu@192.222.56.144:~/data/
scp -i ~/.ssh/freya.pem /work/lambda/ut5-ep19-injected-10p.tgz ubuntu@192.222.56.144:~/data/
scp -i ~/.ssh/freya.pem /work/lambda/ut5-ep19-injected-30p.tgz ubuntu@192.222.57.1:~/data/


# on Lambda
mkdir ~/git
cd ~/git
mkdir -p uT5-ssc
cd uT5-ssc
tar -xzvf ../uT5-repo-clean.tgz
rm -rf ~/git/uT5-repo-clean.tgz

cd ~/data/
tar -xzvf ut5-ep19-clean.tgz
rm -rf ~/data/ut5-ep19-clean.tgz


cd ~/git/uT5-ssc
python3 -m venv ./.venv
source ./.venv/bin/activate
pip install -r requirements.txt
wandb login 647dd79b7e09bbee2824b0d06ec9ece9a9cbba66 --relogin

python -m seq2seq.run_seq2seq configs/5_ut5-ep19-clean/cypher_compact.json