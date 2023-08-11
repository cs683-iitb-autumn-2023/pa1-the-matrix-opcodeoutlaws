make clean
make all
echo "--------------------Matrix size 100--------------------\n" > out.txt
./build/all 100 >> out.txt
echo "--------------------Matrix size 200--------------------\n" >> out.txt
./build/all 200 >> out.txt
echo "--------------------Matrix size 800--------------------\n" >> out.txt
./build/all 800 >> out.txt