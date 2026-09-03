/* class_mapping.h — Official class ID mapping for Track 1 */
#ifndef CLASS_MAPPING_H
#define CLASS_MAPPING_H

#define NUM_CLASSES 17

/* Maps model output index (0–16) to official class ID */
static const int CLASS_ID_MAP[NUM_CLASSES] = {
    0,   /* index 0  → Speed limit 5 km/h   */
    2,   /* index 1  → Speed limit 30 km/h  */
    3,   /* index 2  → Speed limit 40 km/h  */
    4,   /* index 3  → Speed limit 50 km/h  */
    5,   /* index 4  → Speed limit 60 km/h  */
    6,   /* index 5  → Speed limit 70 km/h  */
    7,   /* index 6  → Speed limit 80 km/h  */
    24,  /* index 7  → Go Right             */
    43,  /* index 8  → Go right or straight */
    69,  /* index 9  → Height limit         */
    70,  /* index 10 → Weight limit         */
    72,  /* index 11 → Length limit          */
    83,  /* index 12 → Steep descent        */
    84,  /* index 13 → Steep ascent         */
    85,  /* index 14 → Narrow road          */
    86,  /* index 15 → Narrow bridge        */
    87   /* index 16 → Unknown              */
};

#endif /* CLASS_MAPPING_H */
