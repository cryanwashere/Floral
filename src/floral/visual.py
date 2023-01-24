import graph
from IPython import display

class PositionController(object):
    def __init__(self, left=0, top=0):
        self.left = left
        self.top = top
    def copy(self):
        return PositionController(self.left, self.top)

def make_arrow(x1,y1,x2,y2):
    h = '<svg height="{}" width="{}">'.format(y2 + 20,x2 + 20)
    a = '''
    <defs>
        <marker id="markerArrow" markerWidth="12" markerHeight="12" refX="2" refY="6" orient="auto">
        <path d="M2,2 L2,11 L10,6 L2,2" style="fill: #ffffff;" />
        </marker>
    </defs>
    '''
    b = '<line x1="{}" y1="{}" x2="{}" y2="{}" class="arrow" /></svg>'.format(x1,y1,x2,y2)
    return h + a + b


class VisualizationProbe(object):
    def __init__(self):
        self.document = "<html>"
        self.style = '''
        <style>
            body {
                background-color: black;
                font-family: Arial, Helvetica, sans-serif;
            }
            div {
                position: absolute;
                top: 100px;
                left: 10;
                width: fit-content;
                height: fit-content;
                border: 3px solid #73AD21;
                color: orange;
            }
            .arrow {
                stroke: rgb(255, 255, 255);
                stroke-width: 2;
                marker-end: url(#markerArrow)
            }
            svg {
                position: absolute
            }
        </style>
        '''
        self.document += self.style
    def trace(self, node, position=None):

        if position is None:
            position = PositionController()

        
        
        tag_style = "top: {}px; left: {};".format(position.top, position.left)

        div = '<div style="{}">'.format(tag_style)
        node_tag = "<h3>" + str(node).replace("<","&lt;").replace(">","&gt;") + "</h3>"
        node_tag = div + node_tag + "</div>"
        self.document += node_tag

        position_copy = position.copy()
        position_copy.top += 100
        for i, parent in enumerate(node.parents):
            arrow = make_arrow(
                position.left,
                position.top,
                position_copy.left,
                position_copy.top
            )
            self.document += arrow
            self.trace(parent, position_copy)
            position_copy.left += 400

        
        
    def display(self):
        self.document += "</html>"
        #display.HTML(self.document)
        return self.document

def summary(node):
    visualization_probe = VisualizationProbe()
    visualization_probe.trace(node)
    
    with open("graph_visualizations/out.html","w") as f:
        f.write(visualization_probe.display())
