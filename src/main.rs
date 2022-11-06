/*
 * Copyright 2022 Jannik Birk
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

use bytes::Buf;
use std::collections::{BTreeMap, BTreeSet};
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

impl std::fmt::Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Self::Reg(r) => {
                write!(f, "{:?}", r)?;
            }
            Self::Indexed(off, reg) => {
                write!(f, "{:}({:?})", off as i16, reg)?;
            }
            Self::Immediate(imm) => {
                write!(f, "#{:#x}", imm)?;
            }
            Self::IndexedInc(reg) => {
                write!(f, "@{:?}+", reg)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Instruction {
    Unknown(u16),
    MOV(Operand, Operand),
    MOVB(Operand, Operand),
    ADD(Operand, Operand),
    ADDB,
    ADDC(Operand, Operand),
    ADDCB,
    SUBC(Operand, Operand),
    SUBCB,
    SUB(Operand, Operand),
    SUBB,
    CMP(Operand, Operand),
    CMPB,
    DADD(Operand, Operand),
    DADDB,
    BIT(Operand, Operand),
    BITB,
    BIC(Operand, Operand),
    BICB,
    BIS(Operand, Operand),
    BISB,
    XOR(Operand, Operand),
    XORB,
    AND(Operand, Operand),
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

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Instruction::JNE(off) => {
                write!(f, "JNE #{:x}", off as i16)?;
            }
            Instruction::JNZ(off) => {
                write!(f, "JNZ #{:x}", off as i16)?;
            }
            Instruction::JEQ(off) => {
                write!(f, "JEQ #{:x}", off as i16)?;
            }
            Instruction::JZ(off) => {
                write!(f, "JZ #{:x}", off as i16)?;
            }
            Instruction::JNC(off) => {
                write!(f, "JNC #{:x}", off as i16)?;
            }
            Instruction::JC(off) => {
                write!(f, "JC #{:x}", off as i16)?;
            }
            Instruction::JN(off) => {
                write!(f, "JN #{:x}", off as i16)?;
            }
            Instruction::JGE(off) => {
                write!(f, "JGE #{:x}", off as i16)?;
            }
            Instruction::JL(off) => {
                write!(f, "JL #{:x}", off as i16)?;
            }
            Instruction::JMP(off) => {
                write!(f, "JMP #{:x}", off as i16)?;
            }
            Instruction::MOV(a, b) => {
                write!(f, "MOV {}, {}", a, b)?;
            }
            Instruction::MOVB(a, b) => {
                write!(f, "MOV.B {}, {}", a, b)?;
            }
            Instruction::CALL(target) => {
                write!(f, "CALL {}", target)?;
            }
            Instruction::BR(target) => {
                write!(f, "BR {}", target)?;
            }
            Instruction::ADD(a, b) => {
                write!(f, "ADD {}, {}", a, b)?;
            }
            Instruction::SUB(a, b) => {
                write!(f, "SUB {}, {}", a, b)?;
            }
            Instruction::CMP(a, b) => {
                write!(f, "CMP {}, {}", a, b)?;
            }
            Instruction::BIT(a, b) => {
                write!(f, "BIT {}, {}", a, b)?;
            }
            Instruction::BIS(a, b) => {
                write!(f, "BIS {}, {}", a, b)?;
            }
            Instruction::BIC(a, b) => {
                write!(f, "BIC {}, {}", a, b)?;
            }
            Instruction::XOR(a, b) => {
                write!(f, "XOR {}, {}", a, b)?;
            }
            Instruction::AND(a, b) => {
                write!(f, "AND {}, {}", a, b)?;
            }
            Instruction::ADDC(a, b) => {
                write!(f, "AND {}, {}", a, b)?;
            }
            Instruction::SUBC(a, b) => {
                write!(f, "AND {}, {}", a, b)?;
            }
            Instruction::DADD(a, b) => {
                write!(f, "DADD {}, {}", a, b)?;
            }
            _ => write!(f, "{:?}", self)?,
        }
        Ok(())
    }
}

impl Instruction {
    fn is_branch(self) -> bool {
        self.target().is_some()
    }

    fn is_unconditional(self) -> bool {
        match self {
            Self::JMP(_)
            | Self::CALL(_)
            | Self::RET
            | Self::RETI
            | Self::MOV(_, Operand::Reg(Register::R0))
            | Self::BR(_) => true,
            _ => false,
        }
    }

    fn target(self) -> Option<u16> {
        match self {
            Self::JNE(target)
            | Self::JNZ(target)
            | Self::JEQ(target)
            | Self::JZ(target)
            | Self::JNC(target)
            | Self::JC(target)
            | Self::JN(target)
            | Self::JGE(target)
            | Self::JL(target)
            | Self::JMP(target) => Some(target),

            Self::CALL(operand)
            | Self::MOV(operand, Operand::Reg(Register::R0))
            | Self::BR(operand) => match operand {
                Operand::Immediate(target) => Some(target),
                Operand::Reg(Register::R0) => todo!(),
                Operand::Indexed(_off, Register::R0) => todo!(),
                Operand::IndexedInc(Register::R0) => todo!(),
                _ => None,
            },
            _ => None,
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

    /// Read a u16 from the current memory position and advance the pc.
    fn get_imm(&mut self) -> u16 {
        let imm = self.memory.get_u16_le();
        self.pc += 2;
        imm
    }
}

impl Iterator for MSP430Decoder<'_> {
    type Item = (usize, Instruction);

    fn next(&mut self) -> Option<Self::Item> {
        if self.memory.len() >= 2 {
            let insbegin = self.pc;
            let ins = self.get_imm();

            match ins & 0xf000 {
                0x1000 => match ins & 0x0f80 {
                    0x280 => {
                        let ad = (ins & 0x30) >> 4;
                        let sreg = ins & 0x0f;
                        let operand = match ad {
                            0 => Operand::Reg(sreg.into()),
                            1 => {
                                let offset = self.get_imm();
                                Operand::Indexed(offset, sreg.into())
                            }
                            2 => Operand::Indexed(0, sreg.into()),
                            3 => {
                                if sreg == 0 {
                                    // immediate
                                    let imm = self.get_imm();
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
                0x4000..=0xf000 => {
                    let sreg = (ins & 0xf00) >> 8;
                    let dreg = ins & 0x0f;
                    let ad = (ins & 0x80) >> 7;
                    let _as = (ins & 0x30) >> 4;
                    let isb = (ins & 0x40) != 0;

                    let src = match _as {
                        0 => Operand::Reg(sreg.into()),
                        1 => {
                            let offset = self.get_imm();

                            Operand::Indexed(offset, sreg.into())
                        }
                        2 => Operand::Indexed(0, sreg.into()),
                        3 => {
                            if sreg == 0 {
                                // immediate
                                let imm = self.get_imm();
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
                            let offset = self.get_imm();
                            Operand::Indexed(offset, dreg.into())
                        }
                        _ => panic!(),
                    };

                    match ins & 0xf000 {
                        0x4000 => {
                            if src == Operand::IndexedInc(Register::R1)
                                && dst == Operand::Reg(Register::R0) {
                                Some((insbegin, Instruction::RET))
                            } else {
                                // src operand

                                if dst == Operand::Reg(Register::R0) {
                                    Some((insbegin, Instruction::BR(src)))
                                } else if isb {
                                    Some((insbegin, Instruction::MOVB(src, dst)))
                                } else {
                                    Some((insbegin, Instruction::MOV(src, dst)))
                                }
                            }
                        }
                        0x5000 => Some((insbegin, Instruction::ADD(src,dst))),
                        0x6000 => Some((insbegin, Instruction::ADDC(src,dst))),
                        0x7000 => Some((insbegin, Instruction::SUBC(src,dst))),
                        0x8000 => Some((insbegin, Instruction::SUB(src,dst))),
                        0x9000 => Some((insbegin, Instruction::CMP(src,dst))),
                        0xa000 => Some((insbegin, Instruction::DADD(src,dst))),
                        0xb000 => Some((insbegin, Instruction::BIT(src,dst))),
                        0xc000 => Some((insbegin, Instruction::BIC(src,dst))),
                        0xd000 => Some((insbegin, Instruction::BIS(src,dst))),
                        0xe000 => Some((insbegin, Instruction::XOR(src,dst))),
                        0xf000 => Some((insbegin, Instruction::AND(src,dst))),
                        _ => unreachable!("ins={:x}", ins)
                    }
                }
                _ => Some((insbegin, Instruction::Unknown(ins))),
            }
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BasicBlock {
    id: usize,
    start: usize,
    end: usize,
}

impl BasicBlock {
    fn new(id: usize, start: usize, end: usize) -> Self {
        Self { id, start, end }
    }
}

struct MSP430Analyzer {
    mem: Vec<u8>,
    entry: usize,
    instructions: BTreeMap<usize, Instruction>,
}

impl MSP430Analyzer {
    fn new(mem: Vec<u8>) -> Self {
        let decoder = MSP430Decoder::new(0, &mem);
        let entry = (&mem[0xfffe..]).get_u16_le() as usize;
        let instructions = decoder.collect::<BTreeMap<usize, Instruction>>();

        Self {
            mem,
            entry,
            instructions,
        }
    }

    fn branch_targets(&self) -> Vec<usize> {
        let mut targets = vec![self.entry];

        for (pc, ins) in self.instructions.iter() {
            if ins.is_branch() {
                targets.push(pc + 2);

                if let Some(target) = ins.target() {
                    targets.push(target as usize);
                }
            }
        }

        targets.sort();
        targets
    }

    fn basic_blocks(&self) -> Vec<BasicBlock> {
        let mut ret = Vec::new();

        let targets = self.branch_targets();
        for i in 0..targets.len() - 1 {
            ret.push(BasicBlock::new(i, targets[i], targets[i + 1] - 2));
        }

        ret
    }
}

fn main() {
    let mem = std::fs::read("./memory.bin").unwrap();
    let ana = MSP430Analyzer::new(mem);

    for (pc, ins) in ana.instructions {
        match ins {
            Instruction::Unknown(_) => {}
            _ => println!("{:04x}: {}", pc, ins),
        }
    }
}
