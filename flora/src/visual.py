import graph


class VisualizationProbe(object):
    def trace(self, node):
        print(node)
        for parent in node.parents:
            self.trace(parent)
#stuff
def summary(node):
    visualization_probe = VisualizationProbe()
    visualization_probe.trace(node)