import pathlib
from dataclasses import dataclass
from typing import Optional, Self


@dataclass
class Taxa:
    name: str
    start: Optional[int]
    end: Optional[int]
    replicate: Optional[int]

    @property
    def fasta(self):
        return ">" + self.formatted_name()

    def formatted_name(self):
        return f"{self.name}" + (f"_{self.start}-{self.end}-{self.replicate}"
                                 if self.start is not None
                                 and self.end is not None
                                 and self.replicate is not None
                                 else "")

    def json(self):
        return {
            'taxa': self.name,
            'start': self.start,
            'end': self.end,
            'replicate': self.replicate,
        }

    @staticmethod
    def parse(line) -> Self:
        line_parts = line.lstrip(">").strip().split('_')
        name = line_parts[0]
        if len(line_parts) > 1:
            start, end, replicate = line_parts[1].split('-')
            start, end, replicate = int(start), int(end), int(replicate)
        else:
            start = None
            end = None
            replicate = None
        return Taxa(name, start, end, replicate)


@dataclass
class Sequence:
    taxa: Taxa
    sequence: str

    @staticmethod
    def parse_lines(lines: list[str]) -> Self:
        taxa = Taxa.parse(lines[0])
        seq = ''.join(lines[1:])

        return Sequence(taxa, seq)

    def realign(self, maxsize: int) -> Self:
        prefix = '-' * self.taxa.start
        suffix = '-' * (maxsize - self.taxa.end)
        return Sequence(self.taxa, prefix + self.sequence + suffix)

    def size(self):
        return len(self.sequence)


class Alignment:
    def __init__(self,
                 alignment_filename: Optional[pathlib.Path] = None):
        self._sequences = []
        if alignment_filename is not None:
            with open(alignment_filename) as infile:
                self._parse_alignment(infile)

    def _parse_alignment(self, fp):
        self._sequences = []
        tmp_list = []

        for line in fp:
            if line[0] == ">" and len(tmp_list) != 0:
                self._sequences.append(Sequence.parse_lines(tmp_list))
                tmp_list = []
            tmp_list.append(line.strip())

        self._sequences.append(Sequence.parse_lines(tmp_list))

    def align_to(self, ref_align: Self) -> Self:
        new_size = ref_align.size
        a = Alignment()
        a._sequences = [s.realign(new_size) for s in self._sequences]
        return a

    @property
    def size(self):
        return max([s.size() for s in self._sequences])

    def split_alignment(self, pruned_taxa: set[str]) -> (Self, Self):
        reference_alignment = Alignment()
        query_alignment = Alignment()

        for seq in self._sequences:
            if seq.taxa.name in pruned_taxa:
                query_alignment._sequences.append(seq)
            else:
                reference_alignment._sequences.append(seq)

        return (reference_alignment, query_alignment)

    def write_fasta(self, fp) -> None:
        for seq in self._sequences:
            fp.write(seq.taxa.fasta)
            fp.write("\n")

            fp.write(seq.sequence)
            fp.write("\n")
