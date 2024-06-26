!------------------------------------------------------------------------------
! Flexible Unstructured Simulation Infrastructure with Open Numerics (Open FUSION Toolkit)
!------------------------------------------------------------------------------
!> @file oft_local.F90
!
!> @brief Machine and compiler specific settings.
!!
!! Machine and compiler specific settings and global constants.
!!
!! @author Chris Hansen
!! @date June 2010
!! @ingroup doxy_oft_base
!-----------------------------------------------------------------------------
MODULE oft_local
USE, INTRINSIC :: iso_c_binding, only: c_int, c_ptr, c_long
#ifdef __INTEL_COMPILER
USE ifport ! Intel fortran portability library
#endif
#ifdef HAVE_XML
USE fox_dom, ONLY: fox_node => node, fox_nodelist => nodelist, fox_parsefile => parsefile, &
  fox_getelementsbytagname => getElementsByTagname, fox_item => item, fox_getLength => getLength, &
  fox_extractDataAttribute => extractDataAttribute, fox_hasAttribute => hasAttribute, &
  fox_extractDataContent => extractDataContent, fox_getAttributeNode => getAttributeNode, &
  fox_getExceptionCode => getExceptionCode, fox_DOMException => DOMException
#endif
IMPLICIT NONE
!---Local types sizes
INTEGER, PARAMETER :: i4=SELECTED_INT_KIND(9)           !< 4-Byte integer spec
INTEGER, PARAMETER :: i8=SELECTED_INT_KIND(18)          !< 8-Byte integer spec
INTEGER, PARAMETER :: r4=SELECTED_REAL_KIND(6,37)       !< Single precision float spec (4 Bytes)
INTEGER, PARAMETER :: r8=SELECTED_REAL_KIND(13,307)     !< Double precision float spec (8 Bytes)
INTEGER, PARAMETER :: r10=SELECTED_REAL_KIND(18,4900)   !< Extended precision float spec (10 or 16 Bytes depending on platform)
INTEGER, PARAMETER :: c4=r4                             !< Single precision complex spec (4 Bytes)
INTEGER, PARAMETER :: c8=r8                             !< Double precision complex spec (8 Bytes)
REAL(r8), PARAMETER :: pi=3.141592653589793238462643_r8 !< \f$ \pi \f$
!------------------------------------------------------------------------------
! Define PETSc address type
! - This is the integer value of the C memory pointer to a given object
! and replaces Vec, Mat, IS, etc. I use this instead of the preprocessed
! definitions to keep documentation clean and make the real types obvious.
!------------------------------------------------------------------------------
#ifdef HAVE_PETSC
#if (PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR>5)
#if PETSC_VERSION_MINOR<8
#include "petsc/finclude/petscsysdef.h"
#else
#include "petscconf.h"
#endif
#else
#include "finclude/petscsysdef.h"
#endif
#if (PETSC_SIZEOF_VOID_P == 8)
INTEGER, PARAMETER :: petsc_addr=i8 !< Size of address pointer (32 or 64) bits
#else
INTEGER, PARAMETER :: petsc_addr=i4
#endif
#else
INTEGER, PARAMETER :: petsc_addr=i4
#endif
INTERFACE
!---------------------------------------------------------------------------
!> Interface to C sleep function
!!
!! @result Error code on return
!---------------------------------------------------------------------------
  FUNCTION oft_sleep(seconds)  BIND(C,name="sleep")
    IMPORT c_int
    INTEGER(c_int) :: oft_sleep 
    INTEGER(c_int), INTENT(in), VALUE :: seconds !< Length of time to pause in seconds
  END FUNCTION oft_sleep
!---------------------------------------------------------------------------
!> Simple in-memory hashing function for dataset checksumming
!!
!! @result Checksum of data
!---------------------------------------------------------------------------
  FUNCTION oft_simple_hash(key,length)  BIND(C)
    IMPORT c_int, c_long, c_ptr
    INTEGER(c_int) :: oft_simple_hash
    TYPE(c_ptr), VALUE, INTENT(in) :: key !< Location of data
    INTEGER(c_long), VALUE, INTENT(in) :: length !< Length of data to hash in bytes
  END FUNCTION oft_simple_hash
END INTERFACE
!---------------------------------------------------------------------------
!> One dimensional integer set
!---------------------------------------------------------------------------
TYPE :: oft_1d_int
  INTEGER(i4) :: n = 0 !< Number of values in set
  INTEGER(i4), POINTER, DIMENSION(:) :: v => NULL() !< Values
END TYPE oft_1d_int
!---------------------------------------------------------------------------
!> One dimensional real set
!---------------------------------------------------------------------------
TYPE :: oft_1d_real
  INTEGER(i4) :: n = 0 !< Number of values in set
  REAL(r8), POINTER, DIMENSION(:) :: v => NULL() !< Values
END TYPE oft_1d_real
!---------------------------------------------------------------------------
!> One dimensional complex set
!---------------------------------------------------------------------------
TYPE :: oft_1d_comp
  INTEGER(i4) :: n = 0 !< Number of values in set
  COMPLEX(c8), POINTER, DIMENSION(:) :: v => NULL() !< Values
END TYPE oft_1d_comp
!---------------------------------------------------------------------------
!> Generate inverse of sparse indexing
!---------------------------------------------------------------------------
INTERFACE get_inverse_map
  MODULE PROCEDURE get_inverse_map_i4
  MODULE PROCEDURE get_inverse_map_i8
END INTERFACE get_inverse_map
!
ABSTRACT INTERFACE
!---------------------------------------------------------------------------
!> Generic interface for 1D function
!!
!! @returns Function value
!---------------------------------------------------------------------------
  FUNCTION oft_1d_func(x) result(f)
    IMPORT r8
    REAL(r8), INTENT(in) :: x !< Parameter 1
    REAL(r8) :: f
  END FUNCTION oft_1d_func
!---------------------------------------------------------------------------
!> Generic interface for 2D function
!!
!! @returns Function value
!---------------------------------------------------------------------------
  FUNCTION oft_2d_func(x,y) result(f)
    IMPORT r8
    REAL(r8), INTENT(in) :: x !< Parameter 1
    REAL(r8), INTENT(in) :: y !< Parameter 2
    REAL(r8) :: f
  END FUNCTION oft_2d_func
END INTERFACE
!------------------------------------------------------------------------------
!> Simple timer class
!------------------------------------------------------------------------------
TYPE :: oft_timer
  INTEGER(i8) :: count = 0 !< Integer value of system clock at last call
CONTAINS
  !> Start or reset timer
  procedure :: tick => oft_timer_start
  !> Set elapsed time since tick/tock
  procedure :: tock => oft_timer_elapsed
  !> Get elapsed time since tick/tock in integer counts
  procedure :: int_tock => oft_timer_intelapsed
  !> Check if time since tick/tock exceeds a limit
  procedure :: timeout => oft_timer_timeout
END TYPE oft_timer
PRIVATE oft_timer_start, oft_timer_elapsed, oft_timer_intelapsed, oft_timer_timeout
CONTAINS
!------------------------------------------------------------------------------
!> Start or reset timer
!!
!! @param[in,out] self Calling timer class
!------------------------------------------------------------------------------
SUBROUTINE oft_timer_start(self)
CLASS(oft_timer), INTENT(inout) :: self
self%count=oft_time_i8()
END SUBROUTINE oft_timer_start
!------------------------------------------------------------------------------
!> Set elapsed time since last tick/tock
!!
!! @param[in,out] self Calling timer class
!------------------------------------------------------------------------------
FUNCTION oft_timer_elapsed(self) RESULT(time)
CLASS(oft_timer), INTENT(inout) :: self
INTEGER(i8) :: countnew,crate,cmax,dt
REAL(r8) :: time
CALL system_clock(countnew,crate,cmax)
dt=countnew-self%count
IF(dt<0)dt=dt+cmax
time=dt/REAL(crate,8)
self%count=countnew
END FUNCTION oft_timer_elapsed
!------------------------------------------------------------------------------
!> Get elapsed time since last tick/tock in integer counts
!!
!! @param[in,out] self Calling timer class
!! @return Number of integer counts since last tick/tock
!------------------------------------------------------------------------------
FUNCTION oft_timer_intelapsed(self) result(dt)
CLASS(oft_timer), INTENT(inout) :: self
INTEGER(i8) :: countnew,crate,cmax,dt
CALL system_clock(countnew,crate,cmax)
dt=countnew-self%count
IF(dt<0)dt=dt+cmax
self%count=countnew
END FUNCTION oft_timer_intelapsed
!------------------------------------------------------------------------------
!> Check if time since last tick/tock exceeds a limit
!!
!! @param[in,out] self Calling timer class
!------------------------------------------------------------------------------
FUNCTION oft_timer_timeout(self,timeout) result(test)
CLASS(oft_timer), INTENT(inout) :: self
REAL(r8), INTENT(in) :: timeout !< Length of timeout (seconds)
INTEGER(i8) :: countnew,crate,cmax,dt
REAL(r8) :: time
LOGICAL :: test
CALL system_clock(countnew,crate,cmax)
dt=countnew-self%count
IF(dt<0)dt=dt+cmax
time=dt/REAL(crate,8)
test=(time>timeout)
END FUNCTION oft_timer_timeout
!------------------------------------------------------------------------------
!> Get current system time in integer counts
!!
!! @return System time in integer counts
!------------------------------------------------------------------------------
FUNCTION oft_time_i8() RESULT(time)
INTEGER(i8) :: time,crate,cmax
CALL system_clock(time,crate,cmax)
END FUNCTION oft_time_i8
!------------------------------------------------------------------------------
!> Get elapsed time since a given integer time
!!
!! @param[in,out] self Calling timer class
!! @return Number of integer counts since last tick/tock
!------------------------------------------------------------------------------
FUNCTION oft_time_diff(timein) RESULT(dt)
INTEGER(i8), intent(in) :: timein !< Previous time in integer counts
INTEGER(i8) :: timenew,crate,cmax,dt
CALL system_clock(timenew,crate,cmax)
dt=timenew-timein
IF(dt<0)dt=dt+cmax
END FUNCTION oft_time_diff
!------------------------------------------------------------------------------
!> Skip comment lines in open file
!!
!! @result IOSTAT from last read or -1 if io_unit is not open
!------------------------------------------------------------------------------
function skip_comment_lines(io_unit) result(status)
integer(i4), intent(in) :: io_unit !< I/O unit to advance
integer(i4) :: i,status
logical :: io_open
CHARACTER(LEN=1) :: test_char
INQUIRE(unit=io_unit,opened=io_open)
IF(io_open)THEN
  status=0
  DO WHILE(status==0)
    READ(io_unit,"(A1)",IOSTAT=status,ERR=800)test_char
    BACKSPACE(io_unit,IOSTAT=status,ERR=800)
    IF(test_char=="#")THEN
      READ(io_unit,*,IOSTAT=status,ERR=800)
    ELSE
      RETURN
    END IF
  END DO
ELSE
  status=-1
END IF
800 RETURN
end function skip_comment_lines
!------------------------------------------------------------------------------
!> Get child element within a given XML node
!------------------------------------------------------------------------------
#ifdef HAVE_XML
subroutine xml_get_element(parent,name,element,error_flag,index)
TYPE(fox_node), POINTER, INTENT(in) :: parent
CHARACTER(LEN=*), INTENT(in) :: name
TYPE(fox_node), POINTER, INTENT(out) :: element
INTEGER(i4), INTENT(out) :: error_flag
INTEGER(i4), OPTIONAL, INTENT(out) :: index
INTEGER(i4) :: req_index,nelements
TYPE(fox_nodelist), POINTER :: tmp_list
IF(.NOT.ASSOCIATED(parent))THEN
  error_flag=1
  RETURN
END IF
tmp_list=>fox_getElementsByTagname(parent,TRIM(name))
IF(.NOT.ASSOCIATED(tmp_list))THEN
  error_flag=2
  RETURN
END IF
req_index=1
IF(PRESENT(index))req_index=index
IF(req_index<=0)THEN
  error_flag=3
  RETURN
END IF
nelements=fox_getLength(tmp_list)
IF(nelements==0)THEN
  error_flag=4
  RETURN
END IF
IF(req_index>nelements)THEN
  error_flag=-nelements
  RETURN
END IF
element=>fox_item(tmp_list,req_index-1)
error_flag=0
end subroutine xml_get_element
#endif
!------------------------------------------------------------------------------
!> integer(i4) implementation of \ref oft_local::get_inverse_map
!------------------------------------------------------------------------------
subroutine get_inverse_map_i4(map,n1,imap,n2)
integer(i4), intent(inout) :: map(n1) !< Forward map [n1]
integer(i4), intent(inout) :: imap(n2) !< Inverse map [n2]
integer(i4), intent(in) :: n1 !< Length of forward map
integer(i4), intent(in) :: n2 !< Length of inverse map (n2<=MAX(map))
integer(i4) :: i
! if(size(imap)/=n2)call oft_abort('Invalid array size','get_inverse_map_i4',__FILE__)
imap=0
!$omp parallel do
do i=1,n1
  imap(map(i))=i
end do
end subroutine get_inverse_map_i4
!------------------------------------------------------------------------------
!> integer(i8) implementation of \ref oft_local::get_inverse_map
!------------------------------------------------------------------------------
subroutine get_inverse_map_i8(map,n1,imap,n2)
integer(i8), intent(inout) :: map(n1) !< Forward map [n1]
integer(i4), intent(inout) :: imap(n2) !< Inverse map [n2]
integer(i4), intent(in) :: n1 !< Length of forward map
integer(i8), intent(in) :: n2 !< Length of inverse map (n2<=MAX(map))
integer(i4) :: i
! if(size(imap)/=n2)call oft_abort('Invalid array size','get_inverse_map_i8',__FILE__)
imap=0
!$omp parallel do
do i=1,n1
  imap(map(i))=i
end do
end subroutine get_inverse_map_i8
!------------------------------------------------------------------------------
!> Compute the cross product of two 3 dimensional vectors
!!
!! @result \f$ a \times b \f$ [3]
!------------------------------------------------------------------------------
PURE FUNCTION cross_product(a,b) RESULT(c)
REAL(r8), INTENT(in) :: a(3) !< Vector 1 [3]
REAL(r8), INTENT(in) :: b(3) !< Vector 2 [3]
REAL(r8) :: c(3)
INTEGER(i4), PARAMETER :: i2(3)=[2,3,1],i3(3)=[3,1,2]
c=a(i2)*b(i3)-a(i3)*b(i2)
END FUNCTION cross_product
!------------------------------------------------------------------------------
!> Compute the 2-norm of an array
!!
!! @result \f$ \sum_i a^2_i \f$
!------------------------------------------------------------------------------
PURE FUNCTION magnitude(a) RESULT(c)
REAL(r8), INTENT(in) :: a(:) !< Array
REAL(r8) :: c
c=SQRT(SUM(a**2))
END FUNCTION magnitude
!------------------------------------------------------------------------------
!> Compute the 2-norm of an array
!!
!! @result \f$ \sum_i a^2_i \f$
!------------------------------------------------------------------------------
PURE FUNCTION time_to_string(a) RESULT(c)
REAL(r8), INTENT(in) :: a !< Array
INTEGER(4) :: hours,minutes,seconds
CHARACTER(LEN=13) :: c
hours = FLOOR(a/3600.d0)
minutes = FLOOR((a-hours*3600.d0)/60.d0)
seconds = FLOOR((a-hours*3600.d0-minutes*60.d0))
IF(hours>0)THEN
  WRITE(c,'(I4,A,I2,A,I2,A)')hours,'h ',minutes,'m ',seconds,'s'
ELSE IF(minutes>0)THEN
  WRITE(c,'(I2,A,I2,A,6X)')minutes,'m ',seconds,'s'
ELSE
  WRITE(c,'(I2,A,10X)')seconds,'s'
END IF
END FUNCTION time_to_string
END MODULE oft_local
