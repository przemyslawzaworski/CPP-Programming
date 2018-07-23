// Audio functions.

float bass(float t,float n,float e)   //time,15.0,18.0
{
	float x = fmod(t,0.5f);
	return 0.5f * max(-1.,min(1.,sin(x*pow(2.,((30.-x*5.f-70.)/n))*440.f*6.28f)*10.f*exp(-x*e)));
}
