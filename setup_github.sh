git init .
git add .
git commit -a -m "initial commit"
git clone $1
git remote add origin $1
git push -u origin master