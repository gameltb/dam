import inspect
import mistune

print(inspect.getfullargspec(mistune.HTMLRenderer.image))
