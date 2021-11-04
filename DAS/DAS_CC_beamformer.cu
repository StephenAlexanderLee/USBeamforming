
__constant__ unsigned int recon_nb_lines;
__constant__ unsigned int recon_nb_frames;
__constant__ unsigned int recon_nb_px;
__constant__ float recon_Zres;
__constant__ float recon_Xres;
__constant__ float recon_dtheta;
__constant__ float recon_startdepth;
__constant__ float recon_startwidth;
__constant__ float recon_startangle;
__constant__ unsigned int firstelement;
__constant__ unsigned int lastelement;
__constant__ unsigned int nb_samples;
__constant__ unsigned int nb_angles;
__constant__ float elementpos[256];
__constant__ float lens;
__constant__ float ppw;
__constant__ float Fnumber;

__global__
void DAS_CC_beamform(float* image_out, const float* raw_data)
{
  for (unsigned int iframe = blockIdx.x * blockDim.x + threadIdx.x; iframe < recon_nb_frames; iframe += blockDim.x*gridDim.x) {
    for (unsigned int idx = blockIdx.y * blockDim.y + threadIdx.y; idx < recon_nb_lines; idx += blockDim.y*gridDim.y) {
      for (unsigned int idz = blockIdx.z * blockDim.z + threadIdx.z; idz < recon_nb_px; idz += blockDim.z*gridDim.z) {
        float X = idx * recon_Xres + recon_startwidth;
				float Z = idz * recon_Zres + recon_startdepth;
        float a = Z / Fnumber / 2;
        int apod;
        for (unsigned int ida = 0; ida < nb_angles; ida++){
          float theta = ida * recon_dtheta + recon_startangle;
          float angle_delay = abs(sin(theta)*recon_startwidth);
          for (int element = firstelement; element < lastelement; element++) {
            int index_buff = nb_samples*(nb_angles*(recon_nb_frames*element+iframe)+ida);

            float forward_delay = Z*cosf(theta)+X*sinf(theta);
						float backward_delay = sqrt((X - elementpos[element])*(X - elementpos[element]) + Z * Z);
            float lens_delay = 2 * lens;
            float offset_delay = 2 * recon_startdepth;

            float total_delay_samp = (forward_delay + backward_delay + angle_delay + lens_delay - offset_delay)*ppw;
            if (total_delay_samp < 1.0){total_delay_samp = 1.0;}
            int final_delay = floor(total_delay_samp);
            float alpha = total_delay_samp - float(final_delay);

            if ((elementpos[element] > X - a) && (elementpos[element] < X + a) && final_delay < nb_samples-1) {
              apod = 1;
            }
            else {
              apod = 0;
            }

            //image_out[iframe*recon_nb_lines*recon_nb_px + idx * recon_nb_px + idz] = total_delay_samp;//(1-alpha)*raw_data[final_delay] + alpha*raw_data[final_delay + 1];
            if (apod == 1){
              image_out[iframe*recon_nb_lines*recon_nb_px + idx * recon_nb_px + idz] += ((1-alpha)*raw_data[final_delay+index_buff]+(alpha)*raw_data[final_delay+1+index_buff]);
            }
          }
        }
      }
    }
  }
}
