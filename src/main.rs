use bytes::Buf;
use std::convert::From;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Register {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

impl From<u16> for Register {
    fn from(item: u16) -> Self {
        return [
            Register::R0,
            Register::R1,
            Register::R2,
            Register::R3,
            Register::R4,
            Register::R5,
            Register::R6,
            Register::R7,
            Register::R8,
            Register::R9,
            Register::R10,
            Register::R11,
            Register::R12,
            Register::R13,
            Register::R14,
            Register::R15,
        ][item as usize];
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operand {
    Reg(Register),
    Indexed(u16, Register),
    IndexedInc(Register),
    Immediate(u16),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Instruction {
    Unknown(u16),
    MOV(Operand, Operand),
    MOVB(Operand, Operand),
    ADD,
    ADDB,
    ADDC,
    ADDCB,
    SUBC,
    SUBCB,
    SUB,
    SUBB,
    CMP,
    CMPB,
    DADD,
    DADDB,
    BIT,
    BITB,
    BIC,
    BICB,
    BIS,
    BISB,
    XOR,
    XORB,
    AND,
    ANDB,
    JNE(u16),
    JNZ(u16),
    JEQ(u16),
    JZ(u16),
    JNC(u16),
    JC(u16),
    JN(u16),
    JGE(u16),
    JL(u16),
    JMP(u16),
    CALL(Operand),
    RET,
    RETI,
    BR(Operand),
}

impl Instruction {
    fn is_branch(self) -> bool {
        match self {
            Self::JNE(_)
            | Self::JNZ(_)
            | Self::JEQ(_)
            | Self::JZ(_)
            | Self::JNC(_)
            | Self::JC(_)
            | Self::JN(_)
            | Self::JGE(_)
            | Self::JL(_)
            | Self::JMP(_)
            | Self::CALL(_)
            | Self::RET
            | Self::RETI
            | Self::MOV(_, Operand::Reg(Register::R0))
            | Self::BR(_) => true,
            _ => false,
        }
    }
}

struct MSP430Decoder<'a> {
    memory: &'a [u8],
    pc: usize,
}

impl<'a> MSP430Decoder<'a> {
    fn new(pc: usize, memory: &'a [u8]) -> Self {
        Self { memory, pc }
    }
}

impl Iterator for MSP430Decoder<'_> {
    type Item = (usize, Instruction);

    fn next(&mut self) -> Option<Self::Item> {
        if self.memory.len() > 2 {
            let ins = self.memory.get_u16_le();
            let insbegin = self.pc;
            self.pc += 2;

            // println!(
            //     "sreg={} ad={} isbyte={} as={} dreg={}",
            //     sreg, ad, isb, _as, dreg
            // );

            match ins & 0xf000 {
                0x1000 => match ins & 0x0f80 {
                    0x280 => {
                        let ad = (ins & 0x30) >> 4;
                        let sreg = ins & 0x0f;
                        let operand = match ad {
                            0 => Operand::Reg(sreg.into()),
                            1 => {
                                let offset = self.memory.get_u16_le();
                                self.pc += 2;
                                Operand::Indexed(offset, sreg.into())
                            }
                            2 => Operand::Indexed(0, sreg.into()),
                            3 => {
                                if sreg == 0 {
                                    // immediate
                                    let imm = self.memory.get_u16_le();
                                    self.pc += 2;
                                    Operand::Immediate(imm)
                                } else {
                                    Operand::IndexedInc(sreg.into())
                                }
                            }
                            _ => panic!(),
                        };
                        Some((insbegin, Instruction::CALL(operand)))
                    }
                    0x300 => Some((insbegin, Instruction::RETI)),
                    _ => Some((insbegin, Instruction::Unknown(ins))),
                },
                0x2000 => {
                    let offset = (((ins & 0x3ff) << 1) as i16) << 5 >> 5;
                    let target = (self.pc as u16).wrapping_add(offset as u16);

                    match ins & 0x0c00 {
                        0x0000 => Some((insbegin, Instruction::JNE(target))),
                        0x0400 => Some((insbegin, Instruction::JEQ(target))),
                        0x0800 => Some((insbegin, Instruction::JNC(target))),
                        0x0c00 => Some((insbegin, Instruction::JC(target))),
                        _ => Some((insbegin, Instruction::Unknown(ins))),
                    }
                }
                0x3000 => {
                    let offset = (((ins & 0x3ff) << 1) as i16) << 5 >> 5;
                    let target = (self.pc as u16).wrapping_add(offset as u16);

                    match ins & 0x0c00 {
                        0x0000 => Some((insbegin, Instruction::JN(target))),
                        0x0400 => Some((insbegin, Instruction::JGE(target))),
                        0x0800 => Some((insbegin, Instruction::JL(target))),
                        0x0c00 => Some((insbegin, Instruction::JMP(target))),
                        _ => Some((insbegin, Instruction::Unknown(ins))),
                    }
                }
                0x4000 => {
                    // MOV, MOV.B, RET
                    let sreg = (ins & 0xf00) >> 8;
                    let dreg = ins & 0x0f;
                    let ad = (ins & 0x80) >> 7;
                    let _as = (ins & 0x30) >> 4;
                    let isb = (ins & 0x40) != 0;

                    if sreg == 1 && _as == 3 && ad == 0 && dreg == 0 {
                        Some((insbegin, Instruction::RET))
                    } else {
                        // src operand
                        let src = match _as {
                            0 => Operand::Reg(sreg.into()),
                            1 => {
                                let offset = self.memory.get_u16_le();
                                self.pc += 2;

                                Operand::Indexed(offset, sreg.into())
                            }
                            2 => Operand::Indexed(0, sreg.into()),
                            3 => {
                                if sreg == 0 {
                                    // immediate
                                    let imm = self.memory.get_u16_le();
                                    self.pc += 2;

                                    Operand::Immediate(imm)
                                } else {
                                    Operand::IndexedInc(sreg.into())
                                }
                            }
                            _ => panic!(),
                        };

                        let dst = match ad {
                            0 => Operand::Reg(dreg.into()),
                            1 => {
                                let offset = self.memory.get_u16_le();
                                self.pc += 2;
                                Operand::Indexed(offset, dreg.into())
                            }
                            _ => panic!(),
                        };

                        if dst == Operand::Reg(Register::R0) {
                            Some((insbegin, Instruction::BR(src)))
                        } else if isb {
                            Some((insbegin, Instruction::MOVB(src, dst)))
                        } else {
                            Some((insbegin, Instruction::MOV(src, dst)))
                        }
                    }
                }
                0x5000 => Some((insbegin, Instruction::ADD)),
                0x6000 => Some((insbegin, Instruction::ADDC)),
                0x7000 => Some((insbegin, Instruction::SUBC)),
                0x8000 => Some((insbegin, Instruction::SUB)),
                0x9000 => Some((insbegin, Instruction::CMP)),
                0xa000 => Some((insbegin, Instruction::DADD)),
                0xb000 => Some((insbegin, Instruction::BIT)),
                0xc000 => Some((insbegin, Instruction::BIC)),
                0xd000 => Some((insbegin, Instruction::BIS)),
                0xe000 => Some((insbegin, Instruction::XOR)),
                0xf000 => Some((insbegin, Instruction::AND)),
                _ => Some((insbegin, Instruction::Unknown(ins))),
            }
        } else {
            None
        }
    }
}

fn main() {
    let mem = std::fs::read("./memory.bin").unwrap();
    let decoder = MSP430Decoder::new(0, &mem);

    for (addr, ins) in decoder {
        if ins.is_branch() {
            println!("{:x} {:x?}", addr, ins)
        }
    }
}
