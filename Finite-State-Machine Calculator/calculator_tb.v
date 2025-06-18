module t_dff();
  reg clk, s, rs;
  reg signed [4 :0] in
  reg signed [5:0] out;
  wire A_out, B_out;
  calculator add(in, clk,s,rs,out,A_out,B_out);
  initial
    begin
      $dumpfile("dump.vcd");
      $dumpvars(1);
      s=0;
      rs = 0;
      clk = 1;
      #25
      clk = 0;
      #25
      clk = 1;
      #12
      s = 0;
      rs = 1;
      #10
      in = 5'b000001;
      #3
      clk = 0;
      #25
      clk = 1;
      #12
      s = 1;
      #13
      clk = 0;
      #25
      clk = 1;
      #25
      in = 5'b00101;
      clk = 0;
      #25
      clk = 1;
      #12
      s = 0;
      #13
      clk = 0;
      #25
      clk = 1;
      10
      in = 5'b00111;
      #15
      clk = 0;
      #25
      clk = 1;
      #25
      clk = 0;
      s = 1;
      #25
      clk = 1;
      #25
      clk = 0;
      #25
      clk = 1;
    end
endmodule
