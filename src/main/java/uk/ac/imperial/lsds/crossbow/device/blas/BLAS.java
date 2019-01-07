package uk.ac.imperial.lsds.crossbow.device.blas;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import uk.ac.imperial.lsds.crossbow.SystemConf;
import uk.ac.imperial.lsds.crossbow.data.IDataBuffer;
import uk.ac.imperial.lsds.crossbow.types.DataType;

public class BLAS {
	
	private final static Logger log = LogManager.getLogger (BLAS.class);
	
	private static final BLAS blasInstance = new BLAS ();
	
	public static BLAS getInstance () { return blasInstance; }
	
	private HeapMemoryManager manager;
	
	private boolean loaded;
	
	public BLAS () {
		manager = null;
		loaded = false;
	}
	
	public boolean isLoaded () {
		return loaded;
	}
	
	public void init () {
		
		if (! isLoaded()) {
			try {
				String library = String.format("%s/clib-multigpu/libBLAS.so", SystemConf.getInstance().getHomeDirectory());
				System.load (library);
			} catch (final UnsatisfiedLinkError e) {
				System.err.println(e.getMessage());
				System.exit(1);
			}
			loaded = true;
		}
		
		int numberOfBuffers = SystemConf.getInstance().numberOfBuffers();
		manager = new HeapMemoryManager (numberOfBuffers);
		
		/* Init C memory pool */
		int bufferSize = SystemConf.getInstance().getVariableBufferSize();
		init (numberOfBuffers, bufferSize);
	}
	
	private int arraySize (int rows, int columns, DataType type) {
		
		return (rows * columns * type.sizeOf());
	}
	
	private void checkArrayBounds (String array, String method, int expected, int m, int n) {
		
		int size = arraySize (m, n, DataType.FLOAT);
		if (size != expected)
			throw new IllegalStateException 
				(String.format("error: incorrect size of array %s in %s (found %d, expected %d)", array, method, size, expected));
	}
	
	public void inputDataMovementCallback (int ndx, long address, int size) {
		
		manager.inputDataMovementCallback(ndx, address, size);
	}
	
	public void outputDataMovementCallback (int ndx, long address, int size) {
		
		manager.outputDataMovementCallback(ndx, address, size);
	}
	
	/* 
	 * Perform matrix-matrix operation:  C = alpha (A B) + beta C,
	 * 
	 * where
	 *
	 * A is a M x K matrix,
	 * B is a K x N matrix, and
	 * C is a M x N matrix.
	 */
	
	public int sgemm (
			String TransA, 
			String TransB, 
			int M, 
			int N, 
			int K, 
			float alpha, 
			IDataBuffer A, int startA, int endA, int lda, 
			IDataBuffer B, int startB, int endB, int ldb, 
			float beta, 
			IDataBuffer C, int ldc) {
		
		return sgemm (TransA, TransB, M, N, K, alpha, A, startA, endA, lda, B, startB, endB, ldb, beta, C, 0, C.limit(), ldc);
	}
	
	public int sgemm (
			String TransA, 
			String TransB, 
			int M, 
			int N, 
			int K, 
			float alpha, 
			IDataBuffer A, int startA, int endA, int lda, 
			IDataBuffer B, int startB, int endB, int ldb, 
			float beta, 
			IDataBuffer C, int startC, int endC, int ldc) {
		
		int result = 0;
		
		if (! isLoaded())
			throw new IllegalStateException ("error: BLAS library is not loaded");

		/* Check bounds */
		checkArrayBounds ("A", "sgemm", endA - startA, M, K);
		checkArrayBounds ("B", "sgemm", endB - startB, K, N);
		checkArrayBounds ("C", "sgemm", endC - startC, M, N);
		
		if (! SystemConf.getInstance().useDirectBuffers()) {
		
			Integer x = manager.setAndGet (A, startA, endA);
			Integer y = manager.setAndGet (B, startB, endB);
			Integer z = manager.setAndGet (C, startC, endC);

			result = csgemm (
					TransA, 
					TransB, 
					M, 
					N, 
					K, 
					alpha, 
					x.intValue(), lda, 
					y.intValue(), ldb, 
					beta, 
					z.intValue(), ldc);

			manager.free (x);
			manager.free (y);
			manager.free (z);
		
		} else
			result = csgemm (
					TransA, 
					TransB, 
					M, 
					N, 
					K, 
					alpha, 
					A, startA, endA, lda, 
					B, startB, endB, ldb, 
					beta, 
					C, startC, endC, ldc);

		return result;
	}

	/* Perform matrix-vector operation Y = alpha (A X) + beta Y
	 *
	 * where
	 *
	 * A is a M x N matrix,
	 * X is a N vector if TransA == 'N' or a M vector otherwise
	 * Y is a M vector if TransA == 'N' or a N vector otherwise
	 */
	public int sgemv (
			String TransA,  
			int M, 
			int N,  
			float alpha, 
			IDataBuffer A, int start, int end, int lda, 
			IDataBuffer X, int incX, 
			float beta, 
			IDataBuffer Y, int incY) {

		int result = 0;
		
		if (! isLoaded())
			throw new IllegalStateException ("error: BLAS library is not loaded");
		
		/* Check bounds */
		checkArrayBounds ("A", "sgemv", end - start, M, N);
		checkArrayBounds ("X", "sgemv", X.limit(), 1, TransA.equals("N") ? N : M);
		checkArrayBounds ("Y", "sgemv", Y.limit(), 1, TransA.equals("N") ? M : N);
		
		if (! SystemConf.getInstance().useDirectBuffers()) {
			
			Integer a = manager.setAndGet(A, start, end);
			Integer x = manager.setAndGet(X);
			Integer y = manager.setAndGet(Y);

			result = csgemv (
					TransA, 
					M, 
					N, 
					alpha, 
					a.intValue(), lda, 
					x.intValue(), incX, 
					beta, 
					y.intValue(), incY);

			/* Return buffer slots to queue; a, x, and y slots can now be used by another thread */
			manager.free (a);
			manager.free (x);
			manager.free (y);
		
		} else
			result = csgemv (
					TransA, 
					M, 
					N, 
					alpha, 
					A, start, end, lda, 
					X, incX, 
					beta, 
					Y, incY);
		
		return result;
	}
	
	/* 
	 * Compute Y = (alpha X) + (beta Y) 
	 */
	public int saxpby (
			int N,  
			float alpha, 
			IDataBuffer X, int start, int end, int incX, 
			float beta,
			IDataBuffer Y, int incY) {

		int result = 0;
		
		if (! isLoaded())
			throw new IllegalStateException ("error: BLAS library is not loaded");

		/* Check bounds */
		checkArrayBounds ("X", "saxpby", end - start, 1, N);
		checkArrayBounds ("Y", "saxpby", Y.limit(),   1, N);
		
		if (! SystemConf.getInstance().useDirectBuffers()) {
		
			Integer x = manager.setAndGet(X, start, end);
			Integer y = manager.setAndGet(Y);

			result = csaxpby (
					N, 
					alpha, 
					x.intValue(), incX, 
					beta, 
					y.intValue(), incY);

			manager.free (x);
			manager.free (y);
		
		} else
			csaxpby (
					N, 
					alpha, 
					X, start, end, incX, 
					beta, 
					Y, incY);
		
		return result;
	}
	
	/* BLAS JNI functions */
	
	private native int init (int size, int bufferSize);
	public native int destroy ();
	
	private native int csgemm (
			String TransA, 
			String TransB, 
			int M, 
			int N, 
			int K, 
			float alpha, 
			int A, int lda, 
			int B, int ldb, 
			float beta, 
			int C, int ldc);
	
	private native int csgemm (
			String TransA, 
			String TransB, 
			int M, 
			int N, 
			int K, 
			float alpha, 
			IDataBuffer A, int startA, int endA, int lda, 
			IDataBuffer B, int startB, int endB, int ldb, 
			float beta, 
			IDataBuffer C, int startC, int endC, int ldc);
	
	private native int csgemv (
			String TransA,  
			int M, 
			int N, 
			float alpha, 
			int A, int lda, 
			int X, int incX, 
			float beta, 
			int Y, int incY);
	
	private native int csgemv (
			String TransA,  
			int M, 
			int N,  
			float alpha, 
			IDataBuffer A, int start, int end, int lda, 
			IDataBuffer X, int incX, 
			float beta, 
			IDataBuffer Y, int incY);
	
	private native int csaxpby ( 
			int N,  
			float alpha, 
			int X, 
			int incX,
			float beta,
			int Y, 
			int incY);
	
	private native int csaxpby ( 
			int N,  
			float alpha, 
			IDataBuffer X, int start, int end, int incX,
			float beta,
			IDataBuffer Y, int incY);
}
