from __future__ import print_function, division

import numpy
import matplotlib
import matplotlib.image
 

class SlicePlayer(matplotlib.image.AxesImage):
    # noinspection PyUnusedLocal,PyUnusedLocal
    def __init__(self, ax, arr, cmap=None, norm=None, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, \
                 shape=None, filternorm=1, filterrad=4.0, imlim=None, resample=None, url=None, **kwargs):
        assert arr.ndim==3
        self.arr = arr
        
        if not ax._hold:
            ax.cla()

        if norm is not None:
            assert(isinstance(norm, matplotlib.colors.Normalize))
        if aspect is None:
            aspect = matplotlib.rcParams['image.aspect']
        ax.set_aspect(aspect)
        
        matplotlib.image.AxesImage.__init__(self,ax,cmap,norm,interpolation,origin,extent,filternorm=filternorm,filterrad=filterrad,resample=resample, **kwargs)
        
        self.set_data(self.arr[0,:,:])
        self.i = 0
        self.set_alpha(alpha)
        if self.get_clip_path() is None:
            # image does not already have clipping set, clip to axes patch
            self.set_clip_path(ax.patch)
        #if norm is None and shape is None:
        #    im.set_clim(vmin, vmax)
        if vmin is not None or vmax is not None:
            self.set_clim(vmin, vmax)
        #else:
            #self.autoscale_None()
            #self.set_clim(numpy.amin(self.arr), numpy.amax(self.arr))
        self.set_url(url)

        self.set_extent(self.get_extent())

        ax.add_image(self)
        
        self.axes.figure.canvas.mpl_connect('key_press_event', lambda e: self.keypress(e))
        self.axes.figure.canvas.mpl_connect('scroll_event', lambda e: self.scroll(e))
        
        self.axes.format_coord = lambda x,y:self.format_coord(x,y)
        
    def update_slice(self, i, event=None):
        if i<0: i=0
        elif i>=self.arr.shape[0]: i=self.arr.shape[0]-1
        
        if i!=self.i:
            self.set_data(self.arr[i,:,:])
            self.axes.draw_artist(self)
            self.axes.figure.canvas.blit(self.axes.bbox)
            self.axes.figure.canvas.flush_events()
            self.i = i
            
            if event is not None:
                self.axes.figure.canvas.toolbar.set_message(self.format_coord(event.xdata,event.ydata))
            else:
                self.axes.figure.canvas.toolbar.set_message('slice:{:>6d}'.format(self.i))
            
        return True
        
    def keypress(self, event):
        if event.key=='left':
            return self.update_slice(self.i-1, event)
        elif event.key=='right':
            return self.update_slice(self.i+1, event)
        elif event.key=='shift+left':
            return self.update_slice(self.i-10, event)
        elif event.key=='shift+right':
            return self.update_slice(self.i+10, event)
        else:
            return False
    
    def scroll(self, event):
        if event.button=='down':
            return self.update_slice(self.i-1, event)
        elif event.button=='up':
            return self.update_slice(self.i+1, event)
        else:
            return False
    
    def format_coord(self, x,y):
        return 'slice:{:>6d}, x:{:>8.2f}, y:{:>8.2f}'.format(self.i, x,y)
