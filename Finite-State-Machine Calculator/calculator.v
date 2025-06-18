module d_flipflop(
input D,clk,reset,
output reg Q);

always @(posedge clk)
begin
    if (~reset) begin
        Q<=0;
    end
    else begin
        Q <= D;
    end
end
endmodule

module calculator(
input signed [4 : 0] x,
input clk,s,rs,
output reg [5 :0] y,
output A_out,B_out
);

wire Da,Db;
reg signed [4:0] In1, In2;
reg [1:0] op;

assign Da = A_out&(~B_out) | A_out&s | (~A_out)&(B_out)&(~s);
assign Db = s;

d_flipflop d1(Da,clk,rs,A_out);
d_flipflop d2(Db,clk,rs,B_out);

always @(*)
begin
    case ({A_out,B_out})
        2'b00: begin
            In1 <= 5'b00000;
            In2 <= 5'b00000;
            op <= {x[1],x[0]};
            y <= {x[1],x[0]};
        end
    2'b01: begin
        In1 <= x;
        y <= x;
    end
    2'b10: begin
        In2 <= x;
        y <= x;
    end
    2'b11: begin
        case (op)
            2'b01: y <= In1 + In2;
            2'b10: y <= In1 - In2;
            2'b11: y <= In1*In2;
        endcase
        end
    endcase
end
endmodule
