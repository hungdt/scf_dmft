      subroutine rotate(fin,rot,fout,Nw,Nd)
      implicit none
      integer Nw, Nd
      complex*16 fin(Nw,Nd,Nd), fout(Nw,Nd), rot(Nd,Nd)
Cf2py intent(in) fin,rot,Nw,Nd
Cf2py intent(out) fout

      ! local vars
      integer i,j,k
      complex*16 tmp(Nd,Nd)

!$OMP PARALLEL SHARED(fin,rot,fout) PRIVATE(i,j,tmp)
!$OMP DO
      do i=1,Nw
        call mat_transform(fin(i,:,:), rot, tmp, Nd, 1)
        do j=1,Nd
          fout(i,j) = tmp(j,j);
        enddo
      enddo
!$OMP END PARALLEL
      end


      subroutine irotate(fin,rot,fout,Nw,Nd)
      implicit none
      integer Nw, Nd
      complex*16 fin(Nw,Nd), fout(Nw,Nd,Nd), rot(Nd,Nd)
Cf2py intent(in) fin,rot,Nw,Nd
Cf2py intent(out) fout

      ! local vars
      integer i,j,k
      complex*16 tmp(Nd,Nd)


!$OMP PARALLEL SHARED(fin,rot,fout) PRIVATE(i,j,tmp)
!$OMP DO
      do i=1,Nw
        tmp = 0;
        do j=1,Nd
          tmp(j,j) = fin(i,j);
        enddo
        call mat_transform(tmp, rot, fout(i,:,:), Nd, 0)
      enddo
!$OMP END PARALLEL
      end
        

      subroutine mat_transform(A,B,C,L, direction)
      ! return C = BxAxB^t if direction > 0
      ! return C = B^txAxB if direction = 0
      ! input A a symmetric matrix of size L
      complex*16 A(L,L), B(L,L), C(L,L), tmp(L,L)
      integer L, direction
      character C1, C2
      complex*16 ALPHA,BETA

      ALPHA=1.0; BETA=0.0;

      if (direction > 0) then
          !dsymm to the right: BxA
          C1='N'; C2='N'
          call zgemm(C1,C2,L,L,L,ALPHA,B,L,A,L,BETA,tmp,L)

          !dgemm with B^t: BxAxB^t
          C1='N'; C2='C' ! here is Hermitian conjugate
          call zgemm(C1,C2,L,L,L,ALPHA,tmp,L,B,L,BETA,C,L)
      else 
          !dsymm to the left: AxB
          C1='N'; C2='N'
          call zgemm(C1,C2,L,L,L,ALPHA,A,L,B,L,BETA,tmp,L)

          !dgemm with B^t: B^txAxB
          C1='C'; C2='N' ! Hermitian conjugate
          call zgemm(C1,C2,L,L,L,ALPHA,B,L,tmp,L,BETA,C,L)
      endif
      end
