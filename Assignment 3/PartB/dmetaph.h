#ifndef DMETAPH_H
#define DMETAPH_H

class MString: public CString {
private:
	CString primary;
	CString secondary;
	bool alternate;
	int length;
	int last;
	
public:
	MString();
	MString(const char* in);
	MString(const CString& in);
	bool SlavoGermanic();
	inline void MetaphAdd(const char* main);
	inline void MetaphAdd(const char* main, const char* alt);
	bool IsVowel(int at);
	bool StringAt(int start, int length, ... );
	void DoubleMetaphone(CString &metaph, CString &metaph2);
};

#endif