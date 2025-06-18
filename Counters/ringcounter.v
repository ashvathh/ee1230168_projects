module d \ _flipflop (d , clk ,q , qn );
input d , clk ;
output reg q , qn ;

always @ ( posedge clk )
begin
  case (\{ d ,\~ d \})
    2 ’ b00 : q <= q ;
    2 ’ b01 : q <= 0;
    2 ’ b10 : q <= 1;
    2 ’ b11 : q <= \~ q ;
  endcase
qn <= \~ q ;
end
endmodule

module ringcounter (
input clk ,
output Q0 , Q1 , Q2 , Q3
);
wire Qn0 , Qn1 , Qn2 , Qn3 ;
reg IN ;

d \ _flipflop d0 ( IN , clk , Q0 , Qn0 );
d \ _flipflop d1 ( Q0 , clk , Q1 , Qn1 );
d \ _flipflop d2 ( Q1 , clk , Q2 , Qn2 );
d \ _flipflop d3 ( Q2 , clk , Q3 , Qn3 );

always @ ( posedge clk )
begin
  IN <= Q3 ^ Q2 ;
end
endmodule
