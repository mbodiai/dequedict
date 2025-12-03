from setuptools import Extension, setup

extensions = [
    Extension(
        "dequedict._dequedict",
        sources=["dequedict/_dequedict.c"],
        extra_compile_args=["-O3", "-Wall"],
    ),
]

setup(ext_modules=extensions)

