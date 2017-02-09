# copied from Base
function timelog_sprint(elapsedtime, bytes, gctime, allocs)
    const _mem_units = ["byte", "KB", "MB", "GB", "TB", "PB"]
    const _cnt_units = ["", " k", " M", " G", " T", " P"]
    s = @sprintf "%10.6f seconds"  elapsedtime/1e9
    if bytes != 0 || allocs != 0
        bytes, mb = Base.prettyprint_getunits(bytes, length(_mem_units), Int64(1024))
        allocs, ma = Base.prettyprint_getunits(allocs, length(_cnt_units), Int64(1000))
        if ma == 1
            s *= @sprintf " (%d%s allocation%s: "  allocs  _cnt_units[ma]  allocs==1 ? "" : "s"
        else
            s *= @sprintf " (%.2f%s allocations: "  allocs  _cnt_units[ma]
        end
        if mb == 1
            s *= @sprintf "%d %s%s"  bytes  _mem_units[mb]  bytes==1 ? "" : "s"
        else
            s *= @sprintf "%.3f %s"  bytes  _mem_units[mb]
        end
        if gctime > 0
            s *= @sprintf ", %.2f%% gc time"  100*gctime/elapsedtime
        end
        s *= ")"
    elseif gctime > 0
        s *= @sprintf ", %.2f%% gc time"  100*gctime/elapsedtime
    end
    info(get_logger(current_module()), s)
end

macro timelog(ex)
    quote
        local stats = Base.gc_num()
        local elapsedtime = time_ns()
        local val = $(esc(ex))
        elapsedtime = time_ns() - elapsedtime
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        timelog_sprint(elapsedtime, diff.allocd, diff.total_time, Base.gc_alloc_count(diff))
        val
    end
end
