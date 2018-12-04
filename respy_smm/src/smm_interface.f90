!***************************************************************************************************
!***************************************************************************************************
! TODO: This is a list of small changes that would be nice to make to the core package to allow for easier code in scenarios like this in general.
!
! * It would be nice if file_sim were an optinal argument in the RESPY package then everybody using it from outside does not need to bother with it.
! * How about removing the need that data_sim needs to an allocatable array when passed in to simulate()
!
!
!
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE wrapper_smm(data_sim_int, states_all_int, states_number_period_int, mapping_state_idx_int, max_states_period_int, coeffs_common, coeffs_a, coeffs_b, coeffs_edu, coeffs_home, shocks_cholesky, delta, is_interpolated_int, num_points_interp_int, num_draws_emax_int, num_periods_int, is_myopic_int, is_debug_int, periods_draws_emax_int, num_agents_sim_int, periods_draws_sims, type_spec_shares, type_spec_shifts, edu_start, edu_max, edu_lagged, edu_share, num_paras_int, SLAVECOMM_F2PY)

    !/* external libraries      */

    USE resfort_library

    !/* setup                   */

    IMPLICIT NONE

    !/* external objects        */
    DOUBLE PRECISION, INTENT(OUT)   :: data_sim_int(num_agents_sim_int * num_periods_int, 29)

    INTEGER, INTENT(IN)             :: mapping_state_idx_int(:, :, :, :, :, :)
    INTEGER, INTENT(IN)             :: states_number_period_int(:)
    INTEGER, INTENT(IN)             :: states_all_int(:, :, :)
    INTEGER, INTENT(IN)             :: max_states_period_int
    INTEGER, INTENT(IN)             :: num_points_interp_int
    INTEGER, INTENT(IN)             :: num_agents_sim_int
    INTEGER, INTENT(IN)             :: num_draws_emax_int
    INTEGER, INTENT(IN)             :: num_periods_int
    INTEGER, INTENT(IN)             :: SLAVECOMM_F2PY
    INTEGER, INTENT(IN)             :: num_paras_int
    INTEGER, INTENT(IN)             :: edu_start(:)
    INTEGER, INTENT(IN)             :: edu_max

    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_emax_int(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: periods_draws_sims(:, :, :)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shifts(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: shocks_cholesky(4, 4)
    DOUBLE PRECISION, INTENT(IN)    :: type_spec_shares(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_common(2)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_home(3)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_edu(7)
    DOUBLE PRECISION, INTENT(IN)    :: edu_lagged(:)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_a(15)
    DOUBLE PRECISION, INTENT(IN)    :: coeffs_b(15)
    DOUBLE PRECISION, INTENT(IN)    :: edu_share(:)
    DOUBLE PRECISION, INTENT(IN)    :: delta(1)

    LOGICAL, INTENT(IN)             :: is_interpolated_int
    LOGICAL, INTENT(IN)             :: is_myopic_int
    LOGICAL, INTENT(IN)             :: is_debug_int

    ! TODO: can be removed in the future hopefully
    DOUBLE PRECISION                :: x_all_current(num_paras_int)
    DOUBLE PRECISION, ALLOCATABLE   :: data_sim(:, :)

    INTEGER                         :: num_states
    INTEGER                         :: period

    CHARACTER(225)                  :: file_sim

!---------------------------------------------------------------------------------------------------
! Algorithm
!---------------------------------------------------------------------------------------------------

    ! TODO: remove needed, but requires changes to RESPY as at the beginning of the module.
    file_sim = 'mock.'

    ! Ensure that there is no problem with the repeated allocation of the containers.
    IF (ALLOCATED(periods_rewards_systematic)) DEALLOCATE(periods_rewards_systematic)
    IF (ALLOCATED(states_number_period)) DEALLOCATE(states_number_period)
    IF (ALLOCATED(mapping_state_idx)) DEALLOCATE(mapping_state_idx)
    IF (ALLOCATED(edu_spec%lagged)) DEALLOCATE(edu_spec%lagged)
    IF (ALLOCATED(edu_spec%share)) DEALLOCATE(edu_spec%share)
    IF (ALLOCATED(edu_spec%start)) DEALLOCATE(edu_spec%start)
    IF (ALLOCATED(periods_emax)) DEALLOCATE(periods_emax)
    IF (ALLOCATED(states_all)) DEALLOCATE(states_all)

    !# Transfer global RESFORT variables
    states_all = states_all_int(:, 1:MAXVAL(states_number_period_int), :)
    states_number_period = states_number_period_int
    periods_draws_emax = periods_draws_emax_int
    max_states_period = max_states_period_int
    mapping_state_idx = mapping_state_idx_int
    num_points_interp = num_points_interp_int
    max_states_period = max_states_period_int
    num_types = SIZE(type_spec_shifts, 1)
    is_interpolated = is_interpolated_int
    num_agents_sim = num_agents_sim_int
    num_draws_emax = num_draws_emax_int
    num_periods = num_periods_int
    is_myopic = is_myopic_int
    num_paras = num_paras_int
    is_debug = is_debug_int
    min_idx = edu_max + 1

    optim_paras%shocks_cholesky = shocks_cholesky
    optim_paras%coeffs_common = coeffs_common
    optim_paras%coeffs_home = coeffs_home
    optim_paras%coeffs_edu = coeffs_edu
    optim_paras%coeffs_a = coeffs_a
    optim_paras%coeffs_b = coeffs_b
    optim_paras%delta = delta

    optim_paras%type_shares = type_spec_shares
    optim_paras%type_shifts = type_spec_shifts

    edu_spec%lagged = edu_lagged
    edu_spec%share = edu_share
    edu_spec%start = edu_start
    edu_spec%max = edu_max

    CALL extract_parsing_info(num_paras, num_types, pinfo)

    CALL fort_calculate_rewards_systematic(periods_rewards_systematic, num_periods, states_number_period, states_all, max_states_period, optim_paras)

    IF(SLAVECOMM_F2PY .NE. MISSING_INT) THEN

        ALLOCATE(periods_emax(num_periods, max_states_period))

        CALL MPI_Bcast(2, 1, MPI_INT, MPI_ROOT, SLAVECOMM_F2PY, ierr)

        CALL get_optim_paras(x_all_current, optim_paras, .True.)
        CALL MPI_Bcast(x_all_current, num_paras, MPI_DOUBLE, MPI_ROOT, SLAVECOMM_F2PY, ierr)

        periods_emax = MISSING_FLOAT

        DO period = (num_periods - 1), 0, -1
            num_states = states_number_period(period + 1)
            CALL MPI_RECV(periods_emax(period + 1, :num_states) , num_states, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, SLAVECOMM_F2PY, status, ierr)
        END DO

    ELSE

        CALL fort_backward_induction(periods_emax, num_periods, is_myopic, max_states_period, periods_draws_emax, num_draws_emax, states_number_period, periods_rewards_systematic, mapping_state_idx, states_all, is_debug, is_interpolated, num_points_interp, edu_spec, optim_paras, file_sim, .True.)

    END IF

    CALL fort_simulate(data_sim, periods_rewards_systematic, mapping_state_idx, periods_emax, states_all, num_agents_sim, periods_draws_sims, one_int, file_sim, edu_spec, optim_paras, num_types, is_debug)

    ! TODO: remove needed, but requires changes to RESPY as at the beginning of the module.
    data_sim_int = data_sim

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
