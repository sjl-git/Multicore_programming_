cxx = g++
target = mm
object = src/mm.cc src/main.cc


all : $(target)

$(target) : $(object)
	$(cxx) -o $@ $(object)

clean :
	rm -r -f $(target)
