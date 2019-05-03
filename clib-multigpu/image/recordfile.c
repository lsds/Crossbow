#include "recordfile.h"

#include "../memorymanager.h"

#include "../debug.h"
#include "../utils.h"

#define __HEADER_SIZE sizeof(int)

crossbowRecordFileP crossbowRecordFileCreate (const char *filename, int workers) {
	crossbowRecordFileP p = NULL;
	p = (crossbowRecordFileP) crossbowMalloc (sizeof(crossbow_record_file_t));
	memset (p, 0, sizeof(crossbow_record_file_t));
	p->filename = crossbowStringCopy (filename);
    p->workers = workers;
	crossbowRecordFileOpen (p);
	return p;
}

void crossbowRecordFileOpen (crossbowRecordFileP p) {
    int i;
	nullPointerException (p);
	if (p->opened)
		return;
#ifdef MAP_RECORDS
	p->fd = open(p->filename, O_RDWR);
	if (p->fd < 0) {
		fprintf(stderr, "error: failed to open %s\n", p->filename);
		exit (1);
	}
#else
	p->fp = fopen(p->filename, "rb"); /* Read/binary */
	if (! p->fp) {
		fprintf(stderr, "error: failed to open %s (main)\n", p->filename);
		exit (1);
	}
    /* Allocate a file pointer per worker */
    if (p->workers > 0) {
        p->f = (FILE **) crossbowMalloc (p->workers * sizeof(FILE *));
        for (i = 0; i < p->workers; ++i) {
            p->f[i] = fopen(p->filename, "rb");
	        if (! p->f[i]) {
                fprintf(stderr, "error: failed to open %s (worker #%d)\n", p->filename, i);
                exit (1);
	        }
        }
    }
#endif
	crossbowRecordFileStat (p);
	p->opened = 1;
}

void crossbowRecordFileStat (crossbowRecordFileP p) {
#ifdef MAP_RECORDS
	struct stat sb;
	if (fstat(p->fd, &sb) < 0) {
		fprintf(stderr, "error: failed to stat %s\n", p->filename);
		exit (1);
	}
	p->length = (int) sb.st_size;
#else
	fseek(p->fp, 0L, SEEK_END);
	p->length = (int) ftell (p->fp);
	/* Reset file read pointer */
	rewind(p->fp);
#endif
}

void crossbowRecordFileMap (crossbowRecordFileP p) {
	nullPointerException (p);
#ifdef MAP_RECORDS
	if (p->mapped)
		return;
	p->data = mmap(0, p->length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_NORESERVE, p->fd, 0);
	if (p->data == MAP_FAILED) {
		fprintf(stderr, "error: failed to map %s\n", p->filename);
		exit (1);
	}
	p->mapped = 1;
#endif
	return;
}

void crossbowRecordFileAdviceWillNeed (crossbowRecordFileP p) {
	nullPointerException (p);
#ifdef MAP_RECORDS
	invalidConditionException(p->mapped);
	if (madvise(p->data, p->length, MADV_WILLNEED | MADV_SEQUENTIAL) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	p->needed = 1;
#endif
	return;
}

int crossbowRecordFileHeader (crossbowRecordFileP p) {
	int nr;
	nullPointerException(p);
#ifdef MAP_RECORDS
	illegalStateException ();
#else
	nr = fread(&(p->records), __HEADER_SIZE, 1, p->fp);
	/* Read exactly one integer */
	if (nr != 1) {
		fprintf(stderr, "error: failed to read header from %s\n", p->filename);
		exit (1);
	}
#endif
	return p->records;
}

int crossbowRecordFilePosition (crossbowRecordFileP p) {
	nullPointerException(p);
	invalidConditionException (p->opened);
#ifdef MAP_RECORDS
	invalidConditionException (p->mapped);
	return p->offset;
#else
	return (int) ftell (p->fp);
#endif
}

int crossbowRecordFileRemaining (crossbowRecordFileP p) {
	int remaining;
	remaining = p->length - crossbowRecordFilePosition(p);
	invalidConditionException(remaining >= 0);
	return remaining;
}

unsigned crossbowRecordFileHasRemaining (crossbowRecordFileP p) {
	return (crossbowRecordFileRemaining(p) > 0);
}

void crossbowRecordFileReset (crossbowRecordFileP p, unsigned clear) {
	nullPointerException(p);
	p->counter += 1;
#ifdef MAP_RECORDS
	/* Reset file pointer (don't jump header) */
	p->offset = 0;
	if (clear) {
		/*
		 * If there are more than 1 files in the dataset,
		 * "clear" this file from memory.
		 */
		crossbowRecordFileAdviceWillNotNeed (p);
		crossbowRecordFileUnmap (p);
	}
#else
	(void) clear;
	/* Reset file pointer (don't jump header) */
	rewind(p->fp);
#endif
}

void crossbowRecordFileRead (crossbowRecordFileP p, crossbowRecordP record) {
	nullPointerException (p);
#ifdef MAP_RECORDS
	illegalStateException ();
    if (crossbowRecordFilePosition(p) == 0) /* Skip header */
        p->offset += __HEADER_SIZE;
	/* crossbowRecordReadFromMemory (record, p->data, crossbowRecordFilePosition(p)); */
    /* Explicitly move file pointer */
#else
	if (crossbowRecordFilePosition(p) == 0) /* Skip header */
		fseek (p->fp, __HEADER_SIZE, SEEK_CUR);
	crossbowRecordReadFromFile (record, p->fp, crossbowRecordFilePosition(p));
#endif
	return;
}

int crossbowRecordFileNextPointer (crossbowRecordFileP p) {
    int nr;
    int position;
    int length;
    nullPointerException (p);
#ifdef MAP_RECORDS
    illegalStateException ();
    return 0;
#else
    if (crossbowRecordFilePosition(p) == 0) /* Skip header */
        fseek (p->fp, __HEADER_SIZE, SEEK_CUR);
    
    position = crossbowRecordFilePosition(p);
    
    /* Read record length */
    nr = fread(&length, 4, 1, p->fp);
    invalidConditionException(nr == 1);
    
    fseek(p->fp, position, SEEK_SET);
    fseek(p->fp, length,   SEEK_CUR);
    
    return position;
#endif
}

void crossbowRecordFileReadSafely (crossbowRecordFileP p, int id, int position, crossbowRecordP record) {
    nullPointerException(p);
#ifdef MAP_RECORDS
    illegalStateException ();
#else
    /* info("Read from %p (worker %d) at position %d\n", p->f[id], id, position); */
    nullPointerException (p->f[id]);
    invalidConditionException (position < p->length);
    fseek (p->f[id], position, SEEK_SET);
    crossbowRecordReadFromFile (record, p->f[id], position);
#endif
    return;
}

void crossbowRecordFileAdviceDontNeed (crossbowRecordFileP p) {
	nullPointerException (p);
#ifdef MAP_RECORDS
	invalidConditionException(p->mapped);
	if (! p->needed)
		return;
	if (madvise(p->data, p->length, MADV_DONTNEED) != 0)
		err("Call to madvice() failed: %s\n", strerror(errno));
	p->needed = 0;
#endif
	return;
}

void crossbowRecordFileUnmap (crossbowRecordFileP p) {
	nullPointerException (p);
#ifdef MAP_RECORDS
	if (! p->mapped)
		return;
	munmap (p->data, p->length);
	p->mapped = 0;
#endif
	return;
}

void crossbowRecordFileClose (crossbowRecordFileP p) {
    int i;
	if (! p->opened)
		return;
#ifdef MAP_RECORDS
	close (p->fd);
#else
	fclose (p->fp);
    if (p->f) {
        for (i = 0; i < p->workers; ++i)
            fclose(p->f[i]);
        crossbowFree (p->f, p->workers * sizeof(FILE *));
    }
#endif
	p->opened = 0;
	return;
}

void crossbowRecordFileFree (crossbowRecordFileP p) {
	if (! p)
		return;
	crossbowRecordFileUnmap (p);
	crossbowRecordFileClose (p);
	crossbowStringFree (p->filename);
	crossbowFree (p, sizeof(crossbow_record_file_t));
}
