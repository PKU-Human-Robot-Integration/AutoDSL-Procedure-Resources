import ast
import json


class Statement:
    def __init__(self, opcode="", slot = [], emit="") -> None:
        '''
            A class representing a statement with an opcode, slot, and emit parts.

            @Arguments:

                opcode [str]: The opcode of the statement.
                
                slot [list]: A list representing the slot part of the statement.
                
                emit [str]: The emit part of the statement.

            @Public Methods:

                result() -> str:
                    Returns a string representation of the statement.
                    
                dict() -> dict:
                    Returns a dictionary representation of the statement.
                    
                load_from_dict(dict):
                    Loads the statement from a dictionary representation.
                    
                from_str(s:str) -> Statement:
                    Parses a string to create a Statement object.
        '''

        self.opcode = opcode
        self.slot = slot
        self.emit = emit

    def result(self, format=None) -> str:
        if format is not None:
            slot = json.dumps(self.slot)
            return self.opcode + ": " + slot + " -> " + self.emit
        else:
            return self.opcode + ": " + str(self.slot) + " -> " + self.emit
    
    def dict(self):
        return {
            "opcode": self.opcode,
            "slot": self.slot,
            "emit": self.emit
        }
    
    def load_from_dict(self, dict):
        self.opcode = dict["opcode"]
        self.slot = dict["slot"]
        self.emit = dict["emit"]
        return 
    
    @classmethod
    def from_str(cls, s:str):
        opcode, remainder = s.split(':', 1)
        slot, emit = remainder.split('->', 1)
        return cls(opcode=opcode.strip(), slot=ast.literal_eval(slot.strip()), emit=emit.strip())