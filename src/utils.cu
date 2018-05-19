float randomFloat(float a, float b) {
    return a + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(b-a)));
}
