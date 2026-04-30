echo =======================
for iter in 1 2 3 4 5 6 7 8 9 10
do
echo --- $iter ---
python -u train.py --lr 0.0001 --batch-size 16 --epochs 150 --temp 1 --Dataset 'IEMOCAP' --iemocap-pkl-path 'data/iemocap_multimodal_features_rppg_ses01.pkl' --iemocap-session-prefixes 'Ses01'
done > sdt_iemocap.txt 2>&1 &