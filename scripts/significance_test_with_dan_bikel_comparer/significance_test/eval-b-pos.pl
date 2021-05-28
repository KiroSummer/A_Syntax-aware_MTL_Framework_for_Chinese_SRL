#! /usr/bin/perl


use strict;

if (@ARGV != 3) {
	print "Usage: xxx.pl   pos-gold-file   pos-out-file  evalb-file \n";
	exit(0);
}

open GOLD, $ARGV[0] or die "can't open $ARGV[0] \n";
open TEST, $ARGV[1] or die "can't open $ARGV[1] \n";
open OUT, '>'. $ARGV[2] or die "can't write $ARGV[2] \n";


$/ = "\n\n";
my $punct = 0;
my $token = 0;
my $idx = 1;
my ($sent1, $sent2) = ();

while ( ($sent1 = <GOLD>) and ($sent2 = <TEST>) ) {
	$punct = $token = 0;
	my @sents1 = split "\n", $sent1;
	my @sents2 = split "\n", $sent2;
	if (@sents1 == @sents2) {
		$token = scalar @sents1;
	} else {
		die "tokens number not same!\n";
	}	
	my @pos1 = map { (split" ")[3] } @sents1;
	my @pos2 = map { (split" ")[3] } @sents2;

=cut
	for (@sents1) {
		my $t = (split" ")[3];
		push(@pos1, $t) if $t ne 'PU';
	}

	for (@sents2) {
		my $t = (split" ")[3];
		push(@pos2, $t) if $t ne 'PU';
	}
=cut

	die "non-punct tokens number not same!\n" if @sents1 != @sents2; 

	my $ac = 0;
	for (my $i=0; $i<=$#pos1; $i++) {
		if ($pos1[$i] eq $pos2[$i]) {
			$ac++;
		}
	}
	my $prec = $ac / $token;

	#print OUT $idx, "\t", $token, "\t", "0\t",  $prec, "\t", 
	#	$prec, "\t", $ac, "\t", $ac, "\t", $token-$punct, "\t",
	#	"0\t"x3, "0\n";

	printf OUT "%5d  %5d  0    %6.2f  %6.2f  %5d  %5d  %5d  0 0 0 0\n", $idx, 
		$token, $prec*100, $prec*100, $ac, $ac, $token;

	$idx++;
}
