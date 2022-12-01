# rsync --exclude '.git' -a -P -z -e ssh ./ prakhar@rijuresearch.cse.iitd.ac.in:~/abhinav/
# rsync --exclude '.git' -a -P -z -e ssh ./ nikVision:~/scratch/abhinav/col772a3f/
rsync --exclude '.git' --exclude 'model/cs*.pkl' -a -P -z -e ssh ./ nikVision:~/scratch/abhinav/col772a3_dummy/

# export http_proxy=http://10.10.78.62:3128
# export https_proxy=http://10.10.78.62:3128
# export ftp_proxy=http://10.10.78.62:3128