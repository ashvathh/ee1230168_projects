module jk \ _flipflop (j ,k , clk ,q , qn );
input j ,k , clk ;
output reg q , qn ;

always @ ( posedge clk )
begin
  case (\{ j , k \}) ff
    2 ’ b00 : q <= q ;
    2 ’ b01 : q <= 0;
    2 ’ b10 : q <= 1;
    2 ’ b11 : q <= \~ q ;
  endcase
qn <= \~ q ;
end
endmodule

module graycounter (
input clk ,
output Q0 , Q1 , Q2 , Q3
);
wire Qn0 , Qn1 , Qn2 , Qn3 ;
reg j0 , k0 , j1 , k1 , j2 , k2 , j3 , k3 ;
jk \ _flipflop jk0 ( j0 , k0 , clk , Q0 , Qn0 );
jk \ _flipflop jk1 ( j1 , k1 , clk , Q1 , Qn1 );
jk \ _flipflop jk2 ( j2 , k2 , clk , Q2 , Qn2 );
jk \ _flipflop jk3 ( j3 , k3 , clk , Q3 , Qn3 );

always @ ( posedge clk )
begin
  j0 <= ((\~ Q3 )\&(\~ Q2 )\&(\~ Q1 )) | ((\~ Q3 )\& Q2 \& Q1 ) | ( Q3 \&( Q2 )\&(\~ Q1 )) | ( Q3 \& (\~ Q2 ) \& Q1 );
  k0 <= ((\~ Q3 )\&(\~ Q2 )\& Q1 ) | ((\~ Q3 )\& Q2 \&(\~ Q1 )) | ( Q3 \& Q2 \& Q1 )| ( Q3 \&(\~ Q2 )\&(\~ Q1 ));
  j1 <= ((\~ Q3 )\&(\~ Q2 )\& Q0 ) | ( Q3 \&( Q2 )\& Q0 );
  k1 <= ((\~ Q3 )\& Q2 \& Q0 ) | ( Q3 \&(\~ Q2 )\&( Q0 ));
  j2 <= ((\~ Q3 )\& Q1 \&(\~ Q0 ));
  k2 <= (( Q3 )\&(\~ Q0 )\& Q1 );
  j3 <= ( Q2 \&(\~ Q1 )\&(\~ Q0 ));
  k3 <= ((\~ Q2 )\&(\~ Q1 )\&(\~ Q0 ));
end
endmodule
