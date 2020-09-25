#!/bin/sh

for ((i=351; i<=500; i++))
do
 
	xid=$(head -$i xid_7ms.txt | tail -1)

	cd ..
	cd $xid

	punlearn combine_spectra
	combine_spectra @outsrc.lis 1119_combined bkg_spectra=@outbkg.lis

	cd ../codes
done

