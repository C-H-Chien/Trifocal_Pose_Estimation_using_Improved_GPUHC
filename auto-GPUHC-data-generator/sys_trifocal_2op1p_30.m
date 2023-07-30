function [f, numOfVars] = sys_trifocal_2op1p_30()
    % -- define systems --
    % -- variables --
    syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 x13 x14 x15 x16 x17 x18 x19 x20 x21 x22 x23 x24 x25 x26 x27 x28 x29 x30
    % -- parameters --
    syms p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31 p32 p33
    
    numOfVars = 30;

    f(1) = - p1*p31*x25^2 - 2*p2*p31*x25*x26 - 2*p31*x25*x27 + p1*p31*x26^2 - 2*p31*x26 + p1*p31*x27^2 + 2*p2*p31*x27 - x19 - p1*p31 + p3*x3;
    f(2) = p2*p31*x25^2 - 2*p1*p31*x25*x26 + 2*p31*x25 - p2*p31*x26^2 - 2*p31*x26*x27 + p2*p31*x27^2 - 2*p1*p31*x27 - x20 - p2*p31 + p4*x3;
    f(3) = p31*x25^2 - 2*p1*p31*x25*x27 - 2*p2*p31*x25 + p31*x26^2 - 2*p2*p31*x26*x27 + 2*p1*p31*x26 - p31*x27^2 - p31 + x3 - x21;
    f(4) = - p7*x1*x25^2 - 2*p8*x1*x25*x26 - 2*x1*x25*x27 + p7*x1*x26^2 - 2*x1*x26 + p7*x1*x27^2 + 2*p8*x1*x27 - x19 - p7*x1 + p9*x4;
    f(5) = p8*x1*x25^2 - 2*p7*x1*x25*x26 + 2*x1*x25 - p8*x1*x26^2 - 2*x1*x26*x27 + p8*x1*x27^2 - 2*p7*x1*x27 - x20 - p8*x1 + p10*x4;
    f(6) = x1*x25^2 - 2*p7*x1*x25*x27 - 2*p8*x1*x25 + x1*x26^2 - 2*p8*x1*x26*x27 + 2*p7*x1*x26 - x1*x27^2 - x1 + x4 - x21;
    f(7) = - p13*x2*x25^2 - 2*p14*x2*x25*x26 - 2*x2*x25*x27 + p13*x2*x26^2 - 2*x2*x26 + p13*x2*x27^2 + 2*p14*x2*x27 - x19 - p13*x2 + p15*x5;
    f(8) = p14*x2*x25^2 - 2*p13*x2*x25*x26 + 2*x2*x25 - p14*x2*x26^2 - 2*x2*x26*x27 + p14*x2*x27^2 - 2*p13*x2*x27 - x20 - p14*x2 + p16*x5;
    f(9) = x2*x25^2 - 2*p13*x2*x25*x27 - 2*p14*x2*x25 + x2*x26^2 - 2*p14*x2*x26*x27 + 2*p13*x2*x26 - x2*x27^2 - x2 + x5 - x21;
    f(10) = - p1*p31*x28^2 - 2*p2*p31*x28*x29 - 2*p31*x28*x30 + p1*p31*x29^2 - 2*p31*x29 + p1*p31*x30^2 + 2*p2*p31*x30 - x22 - p1*p31 + p5*x6;
    f(11) = p2*p31*x28^2 - 2*p1*p31*x28*x29 + 2*p31*x28 - p2*p31*x29^2 - 2*p31*x29*x30 + p2*p31*x30^2 - 2*p1*p31*x30 - x23 - p2*p31 + p6*x6;
    f(12) = p31*x28^2 - 2*p1*p31*x28*x30 - 2*p2*p31*x28 + p31*x29^2 - 2*p2*p31*x29*x30 + 2*p1*p31*x29 - p31*x30^2 - p31 + x6 - x24;
    f(13) = - p7*x1*x28^2 - 2*p8*x1*x28*x29 - 2*x1*x28*x30 + p7*x1*x29^2 - 2*x1*x29 + p7*x1*x30^2 + 2*p8*x1*x30 - x22 - p7*x1 + p11*x7;
    f(14) = p8*x1*x28^2 - 2*p7*x1*x28*x29 + 2*x1*x28 - p8*x1*x29^2 - 2*x1*x29*x30 + p8*x1*x30^2 - 2*p7*x1*x30 - x23 - p8*x1 + p12*x7;
    f(15) = x1*x28^2 - 2*p7*x1*x28*x30 - 2*p8*x1*x28 + x1*x29^2 - 2*p8*x1*x29*x30 + 2*p7*x1*x29 - x1*x30^2 - x1 + x7 - x24;
    f(16) = - p13*x2*x28^2 - 2*p14*x2*x28*x29 - 2*x2*x28*x30 + p13*x2*x29^2 - 2*x2*x29 + p13*x2*x30^2 + 2*p14*x2*x30 - x22 - p13*x2 + p17*x8;
    f(17) = p14*x2*x28^2 - 2*p13*x2*x28*x29 + 2*x2*x28 - p14*x2*x29^2 - 2*x2*x29*x30 + p14*x2*x30^2 - 2*p13*x2*x30 - x23 - p14*x2 + p18*x8;
    f(18) = x2*x28^2 - 2*p13*x2*x28*x30 - 2*p14*x2*x28 + x2*x29^2 - 2*p14*x2*x29*x30 + 2*p13*x2*x29 - x2*x30^2 - x2 + x8 - x24;
    f(19) = p3*x9 - p1*p32 - p19*x13 + p21*x14 - 2*p32*x26 - p1*p32*x25^2 + p1*p32*x26^2 + p1*p32*x27^2 - p19*x13*x25^2 + p19*x13*x26^2 + p19*x13*x27^2 + 2*p2*p32*x27 + 2*p20*x13*x27 - 2*p32*x25*x27 - 2*p2*p32*x25*x26 - 2*p20*x13*x25*x26;
    f(20) = p4*x9 - p2*p32 - p20*x13 + p22*x14 + 2*p32*x25 + p2*p32*x25^2 - p2*p32*x26^2 + p2*p32*x27^2 + p20*x13*x25^2 - p20*x13*x26^2 + p20*x13*x27^2 - 2*p1*p32*x27 - 2*p19*x13*x27 - 2*p32*x26*x27 - 2*p1*p32*x25*x26 - 2*p19*x13*x25*x26;
    f(21) = x9 - p32 + p32*x25^2 + p32*x26^2 - p32*x27^2 + 2*p1*p32*x26 - 2*p2*p32*x25 + 2*p19*x13*x26 - 2*p20*x13*x25 - 2*p1*p32*x25*x27 - 2*p2*p32*x26*x27 - 2*p19*x13*x25*x27 - 2*p20*x13*x26*x27;
    f(22) = p9*x11 - p7*p33 - p25*x16 + p27*x17 - 2*p33*x26 - p7*p33*x25^2 + p7*p33*x26^2 + p7*p33*x27^2 - p25*x16*x25^2 + p25*x16*x26^2 + p25*x16*x27^2 + 2*p8*p33*x27 + 2*p26*x16*x27 - 2*p33*x25*x27 - 2*p8*p33*x25*x26 - 2*p26*x16*x25*x26;
    f(23) = p10*x11 - p8*p33 - p26*x16 + p28*x17 + 2*p33*x25 + p8*p33*x25^2 - p8*p33*x26^2 + p8*p33*x27^2 + p26*x16*x25^2 - p26*x16*x26^2 + p26*x16*x27^2 - 2*p7*p33*x27 - 2*p25*x16*x27 - 2*p33*x26*x27 - 2*p7*p33*x25*x26 - 2*p25*x16*x25*x26;
    f(24) = x11 - p33 + p33*x25^2 + p33*x26^2 - p33*x27^2 + 2*p7*p33*x26 - 2*p8*p33*x25 + 2*p25*x16*x26 - 2*p26*x16*x25 - 2*p7*p33*x25*x27 - 2*p8*p33*x26*x27 - 2*p25*x16*x25*x27 - 2*p26*x16*x26*x27;
    f(25) = p5*x10 - p1*p32 - p19*x13 + p23*x15 - 2*p32*x29 - p1*p32*x28^2 + p1*p32*x29^2 + p1*p32*x30^2 - p19*x13*x28^2 + p19*x13*x29^2 + p19*x13*x30^2 + 2*p2*p32*x30 + 2*p20*x13*x30 - 2*p32*x28*x30 - 2*p2*p32*x28*x29 - 2*p20*x13*x28*x29;
    f(26) = p6*x10 - p2*p32 - p20*x13 + p24*x15 + 2*p32*x28 + p2*p32*x28^2 - p2*p32*x29^2 + p2*p32*x30^2 + p20*x13*x28^2 - p20*x13*x29^2 + p20*x13*x30^2 - 2*p1*p32*x30 - 2*p19*x13*x30 - 2*p32*x29*x30 - 2*p1*p32*x28*x29 - 2*p19*x13*x28*x29;
    f(27) = x10 - p32 + p32*x28^2 + p32*x29^2 - p32*x30^2 + 2*p1*p32*x29 - 2*p2*p32*x28 + 2*p19*x13*x29 - 2*p20*x13*x28 - 2*p1*p32*x28*x30 - 2*p2*p32*x29*x30 - 2*p19*x13*x28*x30 - 2*p20*x13*x29*x30;
    f(28) = p11*x12 - p7*p33 - p25*x16 + p29*x18 - 2*p33*x29 - p7*p33*x28^2 + p7*p33*x29^2 + p7*p33*x30^2 - p25*x16*x28^2 + p25*x16*x29^2 + p25*x16*x30^2 + 2*p8*p33*x30 + 2*p26*x16*x30 - 2*p33*x28*x30 - 2*p8*p33*x28*x29 - 2*p26*x16*x28*x29;
    f(29) = p12*x12 - p8*p33 - p26*x16 + p30*x18 + 2*p33*x28 + p8*p33*x28^2 - p8*p33*x29^2 + p8*p33*x30^2 + p26*x16*x28^2 - p26*x16*x29^2 + p26*x16*x30^2 - 2*p7*p33*x30 - 2*p25*x16*x30 - 2*p33*x29*x30 - 2*p7*p33*x28*x29 - 2*p25*x16*x28*x29;
    f(30) = x12 - p33 + p33*x28^2 + p33*x29^2 - p33*x30^2 + 2*p7*p33*x29 - 2*p8*p33*x28 + 2*p25*x16*x29 - 2*p26*x16*x28 - 2*p7*p33*x28*x30 - 2*p8*p33*x29*x30 - 2*p25*x16*x28*x30 - 2*p26*x16*x29*x30;

end


 
