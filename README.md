## lspiv_toolkit
Toolkit for LSPIV implementations

## Code Example

A minimal usage example to create a field and plot it.

```python
from field_toolkit.core.fields import VectorField
from field_toolkit.core.extents import FieldExtents
from field_toolkit.viz.plotting import SimpleFieldView

channelWidth = 100
maxVelocity = 3

domainExtents = FieldExtents.from_bounds_list([0, channelWidth, 0, 50])

field = VectorField.from_developed_pipe_flow_model(channelWidth, maxVelocity, domainExtents)

plot = SimpleFieldView(field, pause=10, autoRefresh=True)
plot.quiver()
```

## Installation

To install run:

```
python setup.py install
```
or 
```
python setup.py develop
```