from copy import deepcopy

class Quartet:
    def __init__(self, s):
        # self.des = s
        self.taxa = []
        s = s.split(",")
        self.taxa.append(s[0][2:]) # ((taxa0
        self.taxa.append(s[1][:-1]) # taxa1)
        self.taxa.append(s[2][1:]) # (taxa2
        if s[3][-1] == ';':
            self.taxa.append(s[3][:-3]) # taxa3));
        else:
            self.taxa.append(s[3][:-2]) # taxa3))

        if self.taxa[0] > self.taxa[1]:
            self.taxa[0], self.taxa[1] = self.taxa[1], self.taxa[0]
        if self.taxa[2] > self.taxa[3]:
            self.taxa[2], self.taxa[3] = self.taxa[3], self.taxa[2]
        if self.taxa[0] > self.taxa[2]:
            self.taxa[0], self.taxa[2] = self.taxa[2], self.taxa[0]
            self.taxa[1], self.taxa[3] = self.taxa[3], self.taxa[1]

        self.des = "((" + self.taxa[0] + "," + self.taxa[1] + "),(" + self.taxa[2] + "," + self.taxa[3] + "));"

    @classmethod
    def from_taxa(cls, t0, t1, t2, t3):
        return cls("((" + t0 + "," + t1 + "),(" + t2 + "," + t3 + "));")
    
    @classmethod
    def from_list(cls, taxa):
        assert len(taxa) == 4
        return cls("((" + taxa[0] + "," + taxa[1] + "),(" + taxa[2] + "," + taxa[3] + "));")
        
    def generate_variants(self):
        variant1 = Quartet("((" + self.taxa[0] + "," + self.taxa[2] + "),(" + self.taxa[1] + "," + self.taxa[3] + "));")
        variant2 = Quartet("((" + self.taxa[0] + "," + self.taxa[3] + "),(" + self.taxa[1] + "," + self.taxa[2] + "));")

        return (variant1, variant2)
        
    def get_first_variant(self):
        copy_taxa = deepcopy(self.taxa)
        copy_taxa.sort()
        return Quartet("((" + copy_taxa[0] + "," + copy_taxa[1] + "),(" + copy_taxa[2] + "," + copy_taxa[3] + "));")

    def __str__(self):
        return self.des

    def __hash__(self):
        return hash(self.des)

    def __eq__(self, other):
        return self.taxa == other.taxa
    
    def __lt__(self, other):
        return self.des < other.des