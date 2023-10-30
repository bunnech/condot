# Perturbation-based Analysis Plan


## Questions to resolve:
1. What is pathogen == Covid & episode == VAP? Is this indeterminate or COVID or VAP?
2. Should we count time to first BAL from admission or intubation


## Preliminary Plan (19.10.2023)

We take samples with: no previous episode and first episode being pneumonia, as 
confirmed by a pathogen test. 

- control = {healthy, NPC}
  - Nick is amazing and gives us labels at the level of samples

- perturbations = {control, pathogen_groups.unique()}
    - first_episode_type = none
    - episode_type is in ['CAP', 'HAP', 'VAP']
    - pathogen_group is not pathogen-negative 
    - (maybe) BAL collected within 72h from ICU admission

- covariates to condition the transport map on = {cell_type, immunocompromised, neutrophiles}

- all covariates we have and not necessarily use:
  - pathogen
  - current episode
  - previous episode
  - (future) future episode
  - first episode
  - clinical resolution
  - (future) superinfection / coinfection
  - (future) time from intubation
  - immunocompromised
  - neutrophiles
  - future resolution
  - future VAP

- things to exclude
    - indeterminate = {everything we excluded}