#!/bin/sh

for ((i=1; i<=1; i++))
do
 
#	bkg=$(head -$i 1119_bkg_1047.txt | tail -1)
#	src=$(head -$i 1119_src_1047.txt | tail -1)
	xid=$(head -$i xid_7ms.txt | tail -1)

	cd ..
	mkdir $xid
	cd $xid

	cp ../codes/bkg_evt.txt .
    cp ../codes/src_evt.txt .
    cp ../codes/outsrc.lis .
    cp ../codes/outbkg.lis .

	mkdir spec
	sed -n "$i"p ../codes/1119_bkg_1047.txt > bkg.reg
	sed -n "$i"p ../codes/1119_src_1047.txt > src.reg

#	echo "$bkg" > bkg.reg
#	echo "$src" > src.reg
	
	punlearn dmextract
	dmextract infile=@src_evt.txt outfile=@outsrc.lis
	dmextract infile=@bkg_evt.txt outfile=@outbkg.lis
	cd ../codes
done

